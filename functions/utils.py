from math import ceil, sqrt, erf

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy import stats


def to_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def visit_to_time(visits):
    if type(visits) is list:
        visits = ['D0' if f.lower() == 'baseline' else f.upper() for f in visits]
    elif type(visits) is str:
        if visits.lower() == 'baseline':
            visits = 'D0'

    if type(visits) is list:
        return [(int(float(f[1:])) if to_float(f[1:]) else np.nan) * (
            7 if f[0] == 'W' else (365 / 12) if f[0] == 'M' else 365 if f[0] == 'Y' else 1) for f in visits]
    if type(visits) is str:
        f = visits
        return (int(float(f[1:])) if to_float(f[1:]) else np.nan) * (
            7 if f[0] == 'W' else (365 / 12) if f[0] == 'M' else 365 if f[0] == 'Y' else 1)


# def time_to_visit(times):
#     if type(times) is list:
#         return [int(f[1:]) * (7 if f[0] == 'W' else (365 / 12) if f[0] == 'M' else 365 if f[0] == 'Y' else 1)
#                 for f in times]
#         return [int(f[1:]) * (7 if f[0] == 'W' else (365 / 12) if f[0] == 'M' else 365 if f[0] == 'Y' else 1)
#                 for f in times]
#         return ['Y' +  if f > 365 else 'M' if f > 30 else 'W' if f > 7 else 'D' for f in times]
#     if type(times) is str:
#         f = times
#         return (int(f[1:]) * (7 if f[0] == 'W' else (365 / 12) if f[0] == 'M' else 365 if f[0] == 'Y' else 1))


def compute_sum(volume):
    volume.groupby(['Time', 'Value'])

    sessions = volume['Session'].unique()
    baseline = list(set([x if x.lower() == 'baseline' else x if (float(x[1:]) if to_float(x[1:])
                                                                 else 1) == 0 else None for x in sessions]))
    if None in baseline:
        baseline.remove(None)

    time = []
    sum = pd.DataFrame([])
    for ses in sessions:
        time.append(volume[volume['Session'] == ses]['Time'].mean() / (365 / 12))
        sum.loc[0, ses] = volume[volume['Session'] == ses]['Value'].sum()

    return time, sum


def compute_evolution(volume, reference='baseline'):
    volume.groupby(['Time', 'Value'])

    sessions = volume['Session'].unique()
    baseline = list(set([x if x.lower() == 'baseline' else x if (float(x[1:]) if to_float(x[1:])
                                                                 else 1) == 0 else None for x in sessions]))
    if None in baseline:
        baseline.remove(None)

    time = []
    evolution = pd.DataFrame([])

    if reference == 'baseline':
        ref = volume[volume['Session'] == baseline[0]]['Value'].sum()
        for ses in sessions:
            time.append(volume[volume['Session'] == ses]['Time'].mean() / (365 / 12))
            evolution.loc[0, ses] = 100 * (volume[volume['Session'] == ses]['Value'].sum() - ref) / ref

    elif reference == 'nadir':
        ref = []
        for ses in sessions:
            ref.append(volume[volume['Session'] == ses]['Value'].sum())
            time.append(volume[volume['Session'] == ses]['Time'].mean() / (365 / 12))
            if np.array(ref).min() == 0:
                evolution.loc[0, ses] = np.inf
            else:
                evolution.loc[0, ses] = 100 * (
                        volume[volume['Session'] == ses]['Value'].sum() - np.array(ref).min()) / np.array(ref).min()

    return time, evolution


def compute_revised_RECIST(volume, lesion):
    idx = [True if x in lesion[lesion].index else False for x in list(volume['VOI'].values)]
    time, evol_bl = compute_evolution(volume[idx], reference='baseline')
    time, evol_nd = compute_evolution(volume[idx], reference='nadir')
    _, vol = compute_sum(volume[idx])
    val = vol.copy().values[0]
    for i in range(1, len(vol.columns)):
        vol[vol.columns[i]] = val[i] - val[:i].min()

    for ses in evol_nd.columns:
        if (evol_nd[ses].values[0] >= 20) & (vol[ses] > 5).any():
            evol_bl[ses] = 'PD'

        elif evol_bl[ses].values[0] <= -30:
            evol_bl[ses] = 'PR'

        elif (evol_bl[ses].values[0] <= -100) & (volume[volume['Session'] == ses]['Value'] < 10).all():
            evol_bl[ses] = 'CR'

        else:
            evol_bl[ses] = 'SD'

    return time, evol_bl


def sample_size_calculation(df, alpha=.05, power=.8, ratio=1, criteria='mST'):
    # from: https://www.statsdirect.com/help/sample_size/survival.htm
    # ref: Dupont WD. Power and sample size calculations. Controlled Clinical Trials 1990;11:116-128.

    ctl = df['Group'] == 0
    exp = df['Group'] == 1

    if criteria == 'HR':
        cph = CoxPHFitter()
        cph.fit(df[['Group', 'Time', 'Event']], duration_col='Time', event_col='Event')

    kmf_ctl = KaplanMeierFitter(alpha=alpha)
    kmf_ctl.fit(df['Time'][ctl], df['Event'][ctl])

    kmf_exp = KaplanMeierFitter(alpha=alpha)
    kmf_exp.fit(df['Time'][exp], df['Event'][exp])

    # POWER: probability of detecting a real effect.
    # ALPHA: probability of detecting a false effect (two sided: double this if you need one sided).
    # A: accrual time during which subjects are recruited to the study.
    # F: additional follow-up time after the end of recruitment.
    # *: input either (C and r) or (C and E), where r=E/C.
    # C: median survival time for control group.
    # E: median survival time for experimental group.
    # r: hazard ratio or ratio of median survival times.
    # M: number of controls per experimental subject.

    A = 12 * 3
    F = 12
    C = kmf_ctl.median_survival_time_
    E = kmf_exp.median_survival_time_
    if criteria == 'HR':
        r = cph.hazard_ratios_.values[0]
    else:
        r = E / C  # cph.hazard_ratios_.values[0]
    M = ratio

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)

    m = (C + E) / 2
    p_a = 1 - np.exp(-np.log(2) * A / m) / (np.log(2) * A / m)
    p = 1 - p_a * np.exp(-np.log(2) * F / m)
    n = (z_a + z_b) ** 2 * (((1 + 1 / m) / p) / (np.log(r) ** 2))

    return ceil(n)


def predictive_probability(df, n_final, alpha=.05):
    # from: https://www.statsdirect.com/help/sample_size/survival.htm
    # ref: Dupont WD. Power and sample size calculations. Controlled Clinical Trials 1990;11:116-128.

    followup_time = df['Time'].max()

    ctl = df['Group'] == 0
    exp = df['Group'] == 1

    kmf_ctl = KaplanMeierFitter(alpha=alpha)
    kmf_ctl.fit(df['Time'][ctl], df['Event'][ctl])

    kmf_exp = KaplanMeierFitter(alpha=alpha)
    kmf_exp.fit(df['Time'][exp], df['Event'][exp])

    cph = CoxPHFitter()
    cph.fit(df[['Group', 'Time', 'Event']], duration_col='Time', event_col='Event')
    # cph.print_summary()

    # Extract parameters
    n = len(df)
    events = df['Event'].sum()
    p_ctl = 1 - kmf_ctl.predict(followup_time)
    p_exp = 1 - kmf_exp.predict(followup_time)

    se = cph.standard_errors_
    lnHR = cph.params_[0]

    def phi(x, m, sd):
        return (1.0 + erf((x - m) / (sd * sqrt(2.0)))) / 2.0

    def phi_HR(n_n=0, x=0):
        n_events = round(n_n * (p_ctl + p_exp) / 2)
        if n_events > 0:
            sd = se * sqrt(events) * sqrt((1 / events) + (1 / n_events))
        else:
            sd = se

        # Cumulative distribution function for the standard normal distribution
        return phi(x, lnHR, sd)

    return phi_HR(n_final - n, x=0)


def power_probability(df, n_final, alpha=.05, condition='PPoS', ratio=1):
    # from: https://arxiv.org/pdf/2102.13550.pdf

    followup_time = df['Time'].max()

    ctl = df['Group'] == 0
    exp = df['Group'] == 1

    kmf_ctl = KaplanMeierFitter(alpha=alpha)
    kmf_ctl.fit(df['Time'][ctl], df['Event'][ctl])

    kmf_exp = KaplanMeierFitter(alpha=alpha)
    kmf_exp.fit(df['Time'][exp], df['Event'][exp])

    cph = CoxPHFitter()
    cph.fit(df[['Group', 'Time', 'Event']], duration_col='Time', event_col='Event')
    # cph.print_summary()

    # Extract parameters
    n = len(df)
    d = df['Event'].sum()
    p_ctl = 1 - kmf_ctl.predict(followup_time)
    p_exp = 1 - kmf_exp.predict(followup_time)
    D = round((n_final - n) * (p_ctl + p_exp) / 2) + d

    delta_d = cph.hazard_ratios_[0]

    r = sqrt((ratio + 1) ** 2 / ratio)
    y = stats.norm.cdf(1 - alpha)

    sigma_0 = (2 / sqrt(d)) ** 2
    Delta_0 = cph.hazard_ratios_[0]
    Delta_1 = 1

    if condition == 'CP':
        return stats.norm.cdf(1 / r * (sqrt(D) * np.log(Delta_1 / delta_d) - r * y) * sqrt(D / (D - d)))
    elif condition == 'PPoS':
        return stats.norm.cdf(1 / r * (sqrt(D) * np.log(Delta_1 / delta_d) - r * y) * sqrt(d / (D - d)))
    else:
        return stats.norm.cdf((sqrt(D) * np.log(Delta_1 / Delta_0) - r * y) / (sqrt(D * sigma_0 ** 2 + r ** 2)))
