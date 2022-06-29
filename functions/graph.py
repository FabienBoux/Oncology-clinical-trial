import datetime
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sn

from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter, WeibullAFTFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, proportional_hazard_test
from matplotlib import pyplot as plt
from scipy.stats import kstest, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from zepid import RiskRatio
from zepid.graphics import EffectMeasurePlot

from clinlib.displaying import Figure
from functions.table import correlation_table
from functions.utils import compute_evolution, compute_revised_RECIST, visit_to_time, power_probability, \
    get_multiparametric_signature


def swimmer_plot(database, followup_time=None, followup_visits=None, metric='Volume', groups=None, groupby=None):
    metadata = database.get_metadata(which='all')

    mt = ['Patient', 'Group', 'Start', 'End', 'Event']
    if 'Expected' in metadata.columns:
        mt.append('Expected')
    if groupby in metadata.columns:
        mt.append(groupby)

    df = metadata[mt].dropna(how='all')

    df['End'] = df['End'].fillna(datetime.datetime.now())
    df['Time'] = (df['End'] - df['Start']).dt.total_seconds() / 3600 / 24 / (365 / 12)

    df = df[~(df['Time'].isna())]

    if groups is None:
        groups = sorted(list(df['Group'].dropna().unique()))

    if followup_time is None:
        followup_time = df['Time'].max()
    else:
        df.loc[df['Time'] > followup_time, 'Time'] = followup_time

    # Plot
    figure = Figure(1)
    figure.set_figsize((1, len(df) / 30))
    ax = figure.get_axes()[0, 0]
    colors = figure.get_colors()

    if groupby in df.columns:
        df = df.sort_values([groupby, 'Time']).reset_index(drop=True)
    else:
        df = df.sort_values(['Time']).reset_index(drop=True)

    if 'Expected' in df.columns:
        df['Expected'][df['Expected'].isna()] = 0
        v = ~df['Expected'].isna()
        y1 = df['Expected'][v]
    else:
        v = ~df['Time'].isna()
        y1 = [np.nan] * v.sum()
    y2 = df['Time'][v]
    patient_ids = df['Patient'][v]

    barWidth = 0.5
    r2 = np.arange(len(y1))
    r1 = np.array([x - barWidth / 4 for x in r2])
    r2 = np.array([x + barWidth / 2 for x in r2])

    if groupby in df.columns:
        jumps = list(Counter(df[groupby]).values())
        jumps.reverse()
        jumps = np.array(jumps).cumsum()
        for i in range(len(jumps) - 1):
            r1[-(jumps[i]):] = r1[-(jumps[i]):] + 1
            r2[-(jumps[i]):] = r2[-(jumps[i]):] + 1

    censored = df['Event'][v].isna()

    for g in range(len(groups)):
        v_group = np.array([True if df['Group'][p] == groups[g] else False for p in df[v].index])

        ax.barh(r2[v_group], y2[v_group], height=barWidth, color=colors[g], alpha=0.65,
                label='Group: {}'.format(groups[g]))
        ax.plot(y2[v_group & censored], r2[v_group & censored], "_", color='black')
        ax.plot(y2[v_group & censored] + followup_time * .01, r2[v_group & censored], ">", color='black')

        if 'Expected' in df.columns:
            ax.barh(r1, y1, height=barWidth / 2, color=['gray'] * len(y1), alpha=0.5,
                    label="Expected survival (if available)")

    patients = database.get_patients(patient_ids)
    for p in range(len(r1)):
        volume = patients[p].get_data(metric)
        lesion = patients[p].get_lesion(metric)

        if followup_visits is not None:
            flwt = followup_visits.copy()
        if (not lesion.empty) & (not volume.empty):
            _, response = compute_revised_RECIST(volume, lesion)

            previous_resp = ''
            for s in response.columns[1:]:
                resp = response[s].values[0]
                t = visit_to_time(s) / (365 / 12)
                if followup_visits is not None:
                    if s in flwt:
                        flwt.remove(s)

                if resp == 'PR':
                    ax.plot(t, r2[p], '.', color='green')
                if resp == 'CR':
                    ax.plot(t, r2[p], 'h', color='green')
                if resp == 'PD':
                    ax.plot(t, r2[p], '*', color='red')

                if ((resp == 'PR') | (resp == 'CR')) & ((previous_resp == 'PR') | (previous_resp == 'CR')):
                    ax.plot([prev_t, t], [r2[p], r2[p]], color='green')
                if (resp == 'PD') & (previous_resp == 'PD'):
                    ax.plot([prev_t, t], [r2[p], r2[p]], color='red')

                previous_resp = resp
                prev_t = t

        if followup_visits is not None:
            if flwt:
                tt = np.array([x for x in visit_to_time(flwt) if (x <= y2[p] * (365 / 12)) & (x > 0)]) / (365 / 12)
                ax.scatter(tt, [r2[p]] * len(tt), color='black', marker='$?$', zorder=20)

    if groupby in df.columns:
        c = 0
        jumps = list(Counter(df[groupby]).values())
        for i in range(len(jumps)):
            ax.plot([-0.2, - 0.2], [c - 0.1, c + jumps[i] - 0.5], color='black')
            plt.text(-0.5, c + 0.4, '{}'.format(df[groupby].unique()[i]), rotation=90, fontsize='medium')
            c = c + jumps[i] + 1

    ax.plot(np.nan, np.nan, ">", color='black', label='Censored patient')
    ax.plot(np.nan, np.nan, '.', color='green', label='Partial Response (PR)')
    ax.plot(np.nan, np.nan, 'h', color='green', label='Complete Response (CR)')
    ax.plot(np.nan, np.nan, color='green', label='Durable response')
    ax.plot(np.nan, np.nan, '*', color='red', label='Progressive Disease (PD)')
    ax.scatter(np.nan, np.nan, color='black', marker='$?$', label='Missing follow-up visit')
    ax.scatter(np.nan, np.nan, color='white',
               label='Absence of response or progression\ncorresponds to a stabilization')

    if followup_visits is not None:
        ax.set_xticks([0] + list(np.array(visit_to_time(followup_visits)) / (365 / 12)))
        ax.set_xticklabels(["{:g} ({})".format(round(a, 1), b) for a, b in
                            zip([0] + list(np.array(visit_to_time(followup_visits)) / (365 / 12)),
                                ['0'] + followup_visits, )])

    ax.set_yticks(r2)
    ax.set_yticklabels(list(df['Patient'][v].values))

    ax.set_xlabel('Time (months) / Follow-up visit')
    ax.set_ylabel('Patients (n = {})'.format(v.sum()))
    ax.set_xlim([ax.get_xlim()[0] - 0.2, followup_time + 0.2])
    ax.set_ylim([r2[0] - barWidth, r2[-1] + barWidth])

    figure.config()
    ax.grid(True, which='major', axis='x', linestyle='--')
    ax.grid(False, axis='y')
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.7))
    return figure


def forest_plot(database, list_metadata, model='HR', followup_time=None, groups=None, n_min=5):
    metadata = database.get_metadata(which='all')

    df = metadata[['Patient', 'Group', 'Start', 'End', 'Event']].dropna(how='all')
    df = pd.concat((df, metadata[list_metadata]), axis=1)

    if groups is None:
        groups = sorted(list(df['Group'].dropna().unique()))
    if len(groups) < 2:
        warnings.warn('Only 1 group provided, required at least 2 groups: ignore figure.')
        return
    elif len(groups) > 2:
        warnings.warn('More than 2 groups provided: figure will only consider the 2 first groups.')
        groups = groups[:2]

    df['End'] = df['End'].fillna(datetime.datetime.now())
    df['Event'] = df['Event'].fillna(0)

    df['Time'] = (df['End'] - df['Start']).dt.total_seconds() / 3600 / 24 / (365 / 12)

    df = df[~(df['Time'].isna())]
    df = df[df['Time'] > 0]

    if followup_time is not None:
        df.loc[df['Time'] > followup_time, 'Time'] = followup_time

    df = df.replace(groups[0], 0).replace(groups[1], 1)

    figure = Figure(1)
    colors = figure.get_colors()
    figure.close()

    # Prepare data according to labels of metadata in 'labs'
    labs = []
    measure = []
    lower = []
    upper = []
    n_rt = []
    n_ax = []
    for i in range(len(list_metadata)):
        cat = df[list_metadata[i]].dropna().unique()

        for j in cat:
            v = df[list_metadata[i]] == j

            if ((df[v]['Group'] == 0).sum() > n_min) & ((df[v]['Group'] == 1).sum() > n_min):
                n_rt.append((df[v]['Group'] == 0).sum())
                n_ax.append((df[v]['Group'] == 1).sum())

                if model == 'HR':
                    cph = CoxPHFitter()
                    cph.fit(df[['Group', 'Time', 'Event']][v], duration_col='Time', event_col='Event')

                    measure.append(round(cph.params_[0], 2))
                    lower.append(round(cph.confidence_intervals_.values[0, 0], 2))
                    upper.append(round(cph.confidence_intervals_.values[0, 1], 2))

                if model == 'RR':
                    rr = RiskRatio()
                    rr.fit(df[['Group', 'Event']][v], exposure='Group', outcome='Event')

                    measure.append(round(rr.results.RiskRatio[1], 2))
                    lower.append(round(rr.results.RR_LCL[1], 2))
                    upper.append(round(rr.results.RR_UCL[1], 2))

                if model == 'TR':
                    weibull_aft = WeibullAFTFitter()
                    weibull_aft.fit(df[['Group', 'Time', 'Event']][v], duration_col='Time', event_col='Event')

                    measure.append(round(weibull_aft.params_[0], 2))
                    lower.append(round(weibull_aft.confidence_intervals_.values[0, 0], 2))
                    upper.append(round(weibull_aft.confidence_intervals_.values[0, 1], 2))

                labs.append("{}: {}   (n={}/{})".format(list_metadata[i], str(j), (df[v]['Group'] == 0).sum(),
                                                        (df[v]['Group'] == 1).sum()))
    if model == 'HR':
        cph = CoxPHFitter()
        cph.fit(df[['Group', 'Time', 'Event']], duration_col='Time', event_col='Event')

        measure.append(round(cph.params_[0], 2))
        lower.append(round(cph.confidence_intervals_.values[0, 0], 2))
        upper.append(round(cph.confidence_intervals_.values[0, 1], 2))

    if model == 'RR':
        rr = RiskRatio()
        rr.fit(df[['Group', 'Event']], exposure='Group', outcome='Event')

        measure.append(round(rr.results.RiskRatio[1], 2))
        lower.append(round(rr.results.RR_LCL[1], 2))
        upper.append(round(rr.results.RR_UCL[1], 2))

    if model == 'TR':
        weibull_aft = WeibullAFTFitter()
        weibull_aft.fit(df[['Group', 'Time', 'Event']], duration_col='Time', event_col='Event')

        measure.append(round(weibull_aft.params_[0], 2))
        lower.append(round(weibull_aft.confidence_intervals_.values[0, 0], 2))
        upper.append(round(weibull_aft.confidence_intervals_.values[0, 1], 2))

    labs.append("Overall (n={}/{})".format((df['Group'] == 0).sum(), (df['Group'] == 1).sum()))
    n_rt.append((df['Group'] == 0).sum())
    n_ax.append((df['Group'] == 1).sum())

    center = (1 if model == 'RR' else 0)
    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(effectmeasure=('log HR' if model == 'HR' else 'log TR' if model == 'TR' else 'RR'), center=center)
    p.colors(pointshape="D", pointcolor=colors[0], errorbarcolor=colors[0])

    x_min = round(max([1.05, (max(upper) * 1.05 if max(upper) > 0 else max(upper) * 0.95)]), 2)
    x_max = round(min([0.95, (min(lower) * 0.95 if min(lower) > 0 else min(lower) * 1.05)]), 2)
    ax = p.plot(figsize=(8, 4), t_adjuster=0.05, max_value=x_min, min_value=x_max)

    if model == 'TR':
        ax.set_xlim(ax.get_xlim()[::-1])

    ax.text(s=u"\u2190 favour {} -".format(groups[1]), x=center, y=1.1 * ax.get_ylim()[1],
            horizontalalignment='right')
    ax.text(s=u"- favour {} \u2192".format(groups[0]), x=center, y=1.1 * ax.get_ylim()[1],
            horizontalalignment='left')

    plt.suptitle("Subgroup", x=-0.1, y=0.98)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

    return plt.gcf()


def volumetry_plot(database, visits=None, which='targets', stat='mean', metric='Volume', groups=None, trendlines=True):
    metadata = database.get_metadata()
    metadata = metadata[['Patient', 'Group']]

    if groups is None:
        groups = sorted(list(metadata['Group'].dropna().unique()))

    figure = Figure(1)
    figure.set_figsize((1.3, 1))
    ax = figure.get_axes()[0, 0]
    colors = figure.get_colors()

    df = pd.DataFrame([])

    patients = database.get_patients()
    for p in range(len(patients)):
        volume = patients[p].get_data(metric)
        d = patients[p].get_data('d')  # TODO: remove these 2 lines or integret them
        volume['Value'] = volume['Value'] * d['Value']
        lesion = patients[p].get_lesion(metric)

        if (not lesion.empty) & (not volume.empty):
            idx = [True if x in lesion[lesion].index else False for x in list(volume['VOI'].values)]
            time, evolution = compute_evolution(volume[idx])

            g = groups.index(metadata[metadata['Patient'] == patients[p].id]['Group'].values[0])
            if not trendlines:
                ax.plot(time, evolution.values[0], '-', color=colors[g], alpha=1 / (len(patients) ** .4))

            evolution.insert(loc=0, column='Group', value=groups[g])
            df = pd.concat((df, evolution))

    if visits is None:
        visits = list(df.columns)
        visits.remove('Group')
    else:
        df = df[['Group'] + [i for i in df.columns if i in visits]]

    # TODO
    # stats.mannwhitneyu()

    for g in range(len(groups)):
        x = np.array(visit_to_time(visits)) / (365 / 12)

        if stat == 'mean':
            y = df[df['Group'] == groups[g]][visits].mean().values
            err = df[df['Group'] == groups[g]][visits].std().values

            if trendlines:
                ax.plot(x, y, color=colors[g], linewidth=2, label="Mean group {}".format(groups[g]))
                plt.fill_between(x, y - err, y + err, color=colors[g], alpha=0.25)
            else:
                ax.plot(x, y, '--', color=colors[g], linewidth=4, label="Mean group {}".format(groups[g]))
                ax.errorbar(x, y, err, fmt='none', color=colors[g], elinewidth=2, capsize=10, capthick=2)

            if g == 1:
                plt.fill_between([np.nan], [np.nan], [np.nan], color='gray', alpha=0.4, label="Standard deviations")

        else:
            y = df[df['Group'] == groups[g]][visits].median().values
            err_low = df[df['Group'] == groups[g]][visits].quantile(.2).values
            err_high = df[df['Group'] == groups[g]][visits].quantile(.8).values

            if trendlines:
                ax.plot(x, y, color=colors[g], linewidth=2, label="Median group {}".format(groups[g]))
                plt.fill_between(x, err_low, err_high, color=colors[g], alpha=0.25)
            else:
                ax.plot(x, y, '--', color=colors[g], linewidth=4, label="Median group {}".format(groups[g]))
                ax.errorbar(x, y, np.vstack([y - err_low, err_high - y]), fmt='none', color=colors[g], elinewidth=2,
                            capsize=10, capthick=2)
            if g == 1:
                plt.fill_between([np.nan], [np.nan], [np.nan], color='gray', alpha=0.4, label="20-80 quantiles")

        # kstest(y, 'norm')
        if g == 0:
            y_ref = y
            val_ref = df[df['Group'] == groups[g]][visits].values.transpose()
        else:
            val = df[df['Group'] == groups[g]][visits].values.transpose()
            diff = 100 * (y_ref - y) / y_ref

            for t in range(len(y)):
                pval = mannwhitneyu(val_ref[t][~np.isnan(val_ref[t])], val[t][~np.isnan(val[t])]).pvalue

                if not np.isnan(diff[t]):
                    plt.text(x[t], y[t] + (8 if diff[t] > 0 else - 8),
                             "{:+.1f}%\n(p={:.2f})".format(diff[t], pval), ha='center', va='center')

    for t in range(len(visits)):
        plt.text(x[t], plt.ylim()[1] + 5, 'n= {}'.format(len(val_ref[t][~np.isnan(val_ref[t])])) + ''.join(
            ['/{}'.format(list(df[['Group', visits[t]]].dropna()['Group'].values).count(gr)) for gr in groups[1:]]),
                 ha='center', va='center')

        ax.set_xticks(x)
        ax.set_xticklabels(["{:g} ({})".format(round(x[i], 1), visits[i]) for i in range(len(visits))])

        ax.set_xlabel('Times (months) / Sessions')
        ax.set_ylabel('Changes in sum of the size of lesions\ncompared to baseline (%)')
        ax.set_xlim([min(x) - 0.5, max(x) + 0.5])

        figure.config()
    return figure


def kaplan_meier_plot(database, event='OS', followup_time=None, cutoff_date=None, groups=None, metric='Volume',
                      adjust_ipfs=False, visits=None):
    cutoff_date = (datetime.datetime.now() if cutoff_date is None
                   else datetime.datetime.strptime(cutoff_date, '%d/%m/%y'))

    metadata = database.get_metadata(which='all')
    df = metadata[['Patient', 'Group', 'Start', 'End', 'Event']].dropna(how='all')

    if groups is None:
        groups = sorted(list(df['Group'].dropna().unique()))
    if len(groups) < 2:
        warnings.warn('Only 1 group, required at least 2 groups: ignore figure.')
        return

    df['End'] = df['End'].fillna(datetime.datetime.now())
    df['Time'] = (df['End'] - df['Start']).dt.total_seconds() / 3600 / 24 / (365 / 12)

    df = df[~(df['Time'].isna())]

    if followup_time is None:
        followup_time = df['Time'].max()
    else:
        df.loc[df['Time'] > followup_time, 'Time'] = followup_time
    # df = df.replace(groups[0], 0).replace(groups[1], 1)

    df['End'] = df['End'].fillna(cutoff_date)
    df['Event'] = df['Event'].fillna(0)

    if event == 'PFS':
        patients = database.get_patients()

        for patient in patients:
            volume = patient.get_data(metric)
            lesion = patient.get_lesion(metric)

            if (not lesion.empty) & (not volume.empty):
                time, response = compute_revised_RECIST(volume, lesion)

                if 'PD' in response.values:
                    t = time[list(response.values[0]).index('PD')]
                    if t < float(df.loc[df['Patient'] == patient.id, 'Time']):
                        df.loc[df['Patient'] == patient.id, 'Time'] = t
                        df.loc[df['Patient'] == patient.id, 'Event'] = 1
                if adjust_ipfs & (visits is not None):
                    if False in [True if i in response.columns else False for i in
                                 visits[:(len(response.columns) - 1)]]:
                        t = time[[True if i in response.columns else False
                                  for i in visits[:(len(response.columns) - 1)]].index(False) + 1]
                        if t < float(df.loc[df['Patient'] == patient.id, 'Time']):
                            df.loc[df['Patient'] == patient.id, 'Time'] = t
                            df.loc[df['Patient'] == patient.id, 'Event'] = 0

    results_all = logrank_test(df['Time'][df['Group'] == groups[0]],
                               df['Time'][df['Group'] == groups[1]],
                               event_observed_A=df['Event'][df['Group'] == groups[0]],
                               event_observed_B=df['Event'][df['Group'] == groups[1]])

    figure = Figure()
    figure.set_figsize((0.8, 0.8))
    colors = figure.get_colors()
    ax = figure.get_axes()[0, 0]

    kmf_model = dict()
    for g in groups:
        v = df['Group'] == g
        kmf_model[g] = KaplanMeierFitter(alpha=0.05)
        kmf_model[g].fit(df['Time'][v], df['Event'][v], label='{} (n={})'.format(g, v.sum()))

        lab = ('>{}'.format(followup_time) if kmf_model[g].median_survival_time_ > followup_time else '{:.2f}'.format(
            kmf_model[g].median_survival_time_))
        leg = 'Group {} (n={}): {}= {}\nMedian follow-up time: {:.2f}'.format(g, v.sum(),
                                                                              ('mPFST' if event == 'PFS' else 'mST'),
                                                                              lab, df['Time'][v].median())
        kmf_model[g].plot_survival_function(ax=ax, ci_show=True, show_censors=True, color=colors[groups.index(g)],
                                            label=leg)
        ax.plot([kmf_model[g].median_survival_time_, kmf_model[g].median_survival_time_], [0, 0.5],
                '--', alpha=0.9, color=colors[groups.index(g)])

    ax.fill_between([np.nan], [np.nan], color="gray", alpha=0.7, label="95% confidence interval")
    ax.plot([0, followup_time], [0.5, 0.5], '--', color="black", alpha=0.9,
            label=('Median PFS time (mPFST)' if event == 'PFS' else 'Median survival time (mST)'))
    ax.plot([np.nan], [np.nan], '.', color='k', label='Censored patient')

    # ax.plot(np.nan, np.nan, '-', color='k',
    #         label='Log-rank test: p={:.2f} (n={})\nwith censored: p={:.2f} (n={})'.format(results.p_value, v.sum(),
    #                                                                                       results_all.p_value,
    #                                                                                       len(rt)))

    df = df[['Group', 'Time', 'Event']].replace('AGuIX', 1).replace('WBRT', 0)
    cph = CoxPHFitter()
    cph.fit(df[['Group', 'Time', 'Event']], duration_col='Time', event_col='Event')

    if proportional_hazard_test(fitted_cox_model=cph, training_df=df[['Group', 'Time', 'Event']],
                                time_transform='log').p_value[0] > 0.05:
        ax.plot(np.nan, np.nan, '-', color='k',
                label='Log-rank test: p={:.2f} (n={})'.format(results_all.p_value, len(v)))
    else:
        ax.plot(np.nan, np.nan, '-', color='k',
                label=' Schoenfeld residual test: p={:.2f} (n={})'.format(
                    proportional_hazard_test(fitted_cox_model=cph, training_df=df[['Group', 'Time', 'Event']],
                                             time_transform='log').p_value[0], len(v)))

    for i in np.arange(0, 3 * len(groups), 3):
        ax.lines[i].set_marker('.')
        ax.lines[i].set_markeredgewidth(0)
        ax.lines[i].set_markersize(12)
        ax.lines[i].set_alpha(0.5)

    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Progression-free survival probability' if event == 'PFS' else 'Survival probability')
    ax.set_xlim([0, followup_time])
    ax.set_ylim([0, 1.05])
    figure.figure.set_figheight(figure.figure.get_figwidth())

    add_at_risk_counts(kmf_model[groups[0]], kmf_model[groups[1]], ax=ax)
    plt.tight_layout()

    figure.config()
    return figure


def response_rate_plot(database, visits=None, criteria='rRECIST', cutoff_date=None, metric='Volume', groups=None):
    cutoff_date = (
        datetime.datetime.now() if cutoff_date is None else datetime.datetime.strptime(cutoff_date, '%d/%m/%y'))

    metadata = database.get_metadata(which='all')

    if groups is None:
        groups = sorted(list(metadata['Group'].dropna().unique()))

    df = pd.DataFrame([])
    patients = database.get_patients()
    for p in range(len(patients)):
        volume = patients[p].get_data(metric)
        lesion = patients[p].get_lesion(metric)
        volume = volume[volume['Study'] < cutoff_date]

        if (not lesion.empty) & (not volume.empty):
            if criteria == 'rRECIST':
                _, response = compute_revised_RECIST(volume, lesion)
            elif criteria == 'mRECIST':
                _, response = compute_revised_RECIST(volume, patients[p].get_lesion(metric, max_number=np.inf))
            else:
                _, response = compute_revised_RECIST(volume, lesion)

            g = groups.index(metadata[metadata['Patient'] == patients[p].id]['Group'].values[0])
            response.insert(loc=0, column='Group', value=groups[g])

            df = pd.concat((df, response))

    if visits is None:
        visits = list(df.columns)
        visits.remove('Group')
    else:
        df = df[['Group'] + [i for i in df.columns if i in visits]]

    figure = Figure(1)
    figure.set_figsize((1.3, 1))
    ax = figure.get_axes()[0, 0]
    colors = figure.get_colors()

    labels = list(df.columns)[1:]
    width = .8
    x = np.arange(len(labels))

    for g in range(len(groups)):
        pr = (df[labels][df['Group'] == groups[g]] == 'PR').sum().values
        cr = (df[labels][df['Group'] == groups[g]] == 'CR').sum().values
        n = (~((df[labels][df['Group'] == groups[g]]).isna())).sum().values

        ax.bar(x - width / 2 + (g + .5) * width / len(groups), 100 * (pr + cr) / n,
               width / len(groups), color=colors[g], label='Group: ' + groups[g])
        ax.bar(x - width / 2 + (g + .5) * width / len(groups), 100 * cr / n,
               width / len(groups), color=colors[g], edgecolor='black', hatch='//')

        for i in np.arange(len(x)):
            ax.text(x[i] - width / 2 + (g + .5) * width / len(groups), 100 * (pr[i] + cr[i]) / n[i] + 2, str(n[i]),
                    ha='center')

    ax.bar(np.nan, np.nan, width, color='white', edgecolor='black', hatch='//', label='CR')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel('Sessions')
    ax.set_ylabel('Percentage of responses (PR or CR)')

    ax.set_ylim([0, ax.get_ylim()[1] + 5])

    figure.config()
    return figure


def probability_of_success_plot(database, n_total, followup_time=None, group=None, event='OS', metric='Volume',
                                visits=None, adjust_ipfs=True):
    metadata = database.get_metadata(which='in')

    df = pd.DataFrame({'Patient': metadata['Patient'].values,
                       'Group': metadata['Group'].values,
                       'Start': metadata['Start'].values,
                       'End': metadata['End'].values,
                       'Event': metadata['Event'].values,
                       }).dropna(how='all')

    if group is None:
        group = df['Group'].unique()
        group = group[:2]
        group = np.array([group[1], group[0]])

    df['End'] = df['End'].fillna(datetime.datetime.now())
    df['Event'] = df['Event'].fillna(0)

    df['Time'] = (df['End'] - df['Start']).dt.total_seconds() / 3600 / 24 / (365 / 12)

    df = df[~(df['Time'].isna())]

    if event == 'PFS':
        patients = database.get_patients()

        for patient in patients:
            volume = patient.get_data(metric)
            lesion = patient.get_lesion(metric)

            if (not lesion.empty) & (not volume.empty):
                time, response = compute_revised_RECIST(volume, lesion)

                if 'PD' in response.values:
                    t = time[list(response.values[0]).index('PD')]
                    if t < float(df.loc[df['Patient'] == patient.id, 'Time']):
                        df.loc[df['Patient'] == patient.id, 'Time'] = t
                        df.loc[df['Patient'] == patient.id, 'Event'] = 1

                if adjust_ipfs & (visits is not None):
                    if False in [True if i in response.columns else False for i in
                                 visits[:(len(response.columns) - 1)]]:
                        t = time[[True if i in response.columns else False
                                  for i in visits[:(len(response.columns) - 1)]].index(False) + 1]
                        if t < float(df.loc[df['Patient'] == patient.id, 'Time']):
                            df.loc[df['Patient'] == patient.id, 'Time'] = t
                            df.loc[df['Patient'] == patient.id, 'Event'] = 0

    if followup_time is not None:
        df.loc[df['Time'] > followup_time, 'Event'] = 0
        df.loc[df['Time'] > followup_time, 'Time'] = followup_time

    df = df.replace(group[0], 0).replace(group[1], 1)

    Dmin = list(np.arange(0, 1.01, .01))

    figure = Figure(1)
    ax = figure.get_axes()[0, 0]

    cp = np.array(power_probability(df, n_total, Dmin=Dmin, condition='CP', success='clinical')) * 100
    ppos = np.array(power_probability(df, n_total, Dmin=Dmin, condition='PPoS', success='clinical')) * 100
    pos = np.array(power_probability(df, n_total, Dmin=Dmin, condition='PoS', success='clinical')) * 100

    # ax.plot(100 * (1 - np.array(Dmin)), cp, label='CP')
    # ax.plot(100 * (1 - np.array(Dmin)), ppos, label='PPoS')
    # ax.plot(100 * (1 - np.array(Dmin)), pos, label='PoS')

    # plt.fill_between(100 * (1 - np.array(Dmin)),
    #                  np.min(np.array((cp, ppos, pos)), axis=0), np.max(np.array((cp, ppos, pos)), axis=0),
    #                  color="gray", alpha=0.25)
    #
    # ax.plot([0, 100], [50, 50], color='black', linewidth=1)
    # ax.plot(np.array([1, 1]) * (1 - Dmin[np.argmin(abs(cp - 50))]) * 100, [0, 50], color='black', linewidth=1)
    #
    # ax.set_xticks(list(ax.get_xticks()) + [(1 - Dmin[np.argmin(abs(cp - 50))]) * 100])
    #
    # ax.set_xlabel('Minimum effect on hazard ratio (%)')
    # ax.set_ylabel('Probability (%)')

    # ax.set_xlim([0, 100])
    # ax.set_ylim([0, 100])

    ax.plot(np.array(Dmin), cp, label='CP')
    ax.plot(np.array(Dmin), ppos, label='PPoS')
    ax.plot(np.array(Dmin), pos, label='PoS')

    plt.fill_between(np.array(Dmin),
                     np.min(np.array((cp, ppos, pos)), axis=0), np.max(np.array((cp, ppos, pos)), axis=0),
                     color="gray", alpha=0.25)

    ax.plot([0, 100], [50, 50], color='black', linewidth=1)
    ax.plot(np.array([1, 1]) * (Dmin[np.argmin(abs(cp - 50))]), [0, 50], color='black', linewidth=1)

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1] + [(Dmin[np.argmin(abs(cp - 50))])])

    ax.set_xlabel('HR')
    ax.set_ylabel('Probability( < HR ) (%)')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])

    figure.config()
    return figure


def correlation_matrix_plot(database, list_data, visit=None, threshold=False):
    tab = correlation_table(database, list_data, visit=visit)

    figure = Figure(1)
    ax = figure.get_axes()[0, 0]

    if not tab.empty:
        if threshold:
            tab = abs(tab)
            tab = 0 * (tab < .25) + .25 * ((.25 <= tab) & (tab < .5)) + .5 * ((.5 <= tab) & (tab < .75)) + .75 * (
                    .75 <= tab)

            sn.heatmap(tab, ax=ax, annot=False, cbar_kws={'label': 'Classification of Pearson correlation (r)'})

            c_bar = ax.collections[0].colorbar
            c_bar.set_ticks([0, .25, .5, .75])
            c_bar.set_ticklabels(['No', 'Weak', 'Moderate', 'Strong'])

        else:
            sn.heatmap(tab, ax=ax, annot=True, fmt=".2", annot_kws={'size': 10}, vmin=-1, vmax=1,
                       cbar_kws={'label': 'Pearson correlation (r)'})

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    else:
        warnings.warn('No data to show.')

    figure.config()
    return figure


def dim_reduction_plot(database, list_data, visit=None, method='PCA', outliers=0, groups=None):
    metadata = database.get_metadata(which='all')
    metadata = metadata[['Patient', 'Group']]
    patients = database.get_patients()

    df = pd.DataFrame([])
    patient_id = np.array([])
    patient_group = np.array([])
    for pat in patients:
        d = get_multiparametric_signature(pat, session=visit, parameters=list_data)
        df = pd.concat((df, d)).reset_index(drop=True)
        patient_id = np.concatenate((patient_id, np.array([pat.id] * len(d))))
        patient_group = np.concatenate(
            (patient_group, np.array([metadata.loc[metadata['Patient'] == pat.id, 'Group'].values[0]] * len(d))))

    data = df.dropna().reset_index(drop=True)
    patient_id = patient_id[~(df.isna().any(axis=1))]
    patient_group = patient_group[~(df.isna().any(axis=1))]
    if (len(data) / len(df)) < .8:
        print("Warning: {}% of data have been removed because containing NaN values.".format(
            round(100 * len(data) / len(df), 1)))
        print("         Consider ignoring the parameters that cause such a reduction.")

    if groups is None:
        groups = sorted(list(set(patient_group)))

    # Dimension reduction
    X = np.array([data.loc[i].values for i in range(len(data))])

    if method == 'PCA':
        pca = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])
        X_embedded = pca.fit_transform(X)

    if method == 't-SNE':
        tsne = Pipeline([('scaling', StandardScaler()), ('tsne', TSNE(n_components=2))])
        X_embedded = tsne.fit_transform(X)

    X_ = np.array([x[0] for x in X_embedded])
    Y_ = np.array([x[1] for x in X_embedded])

    figure = Figure(2, nb_columns=1)
    axs = figure.get_axes().reshape(-1)
    col = figure.get_colors()
    colors = figure.get_colors(len(patients))
    markers = ['o', '^', 'v', '8', 's', 'P', '*', 'h', 'D', 'X']

    for i in range(len(patients)):
        if (patient_id == patients[i].id).any():
            axs[0].scatter(X_[patient_id == patients[i].id], Y_[patient_id == patients[i].id],  # s=size[v]
                           color=colors[i], alpha=0.75, marker=markers[np.random.randint(len(markers))],
                           label=patients[i].id)
    for i in range(len(groups)):
        axs[1].scatter(X_[patient_group == groups[i]], Y_[patient_group == groups[i]],  # s=size[v]
                       color=col[i], alpha=0.75, label=groups[i])

    if outliers > 0:
        # Automatic outlier detection algorithm
        # for details, see: https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
        iso = IsolationForest(contamination=outliers / 100)
        yhat = iso.fit_predict(X)

        axs[0].plot(X_[yhat == -1], Y_[yhat == -1], 'x', color='black', label='Outliers')
        axs[1].plot(X_[yhat == -1], Y_[yhat == -1], 'x', color='black', label='Outliers')

    if method == 'PCA':
        var_explained = pca.get_params()['pca'].explained_variance_ratio_

        axs[1].set_xlabel('Principal component 1 ({:.1f}%)'.format(100 * var_explained[0]))
        axs[0].set_ylabel('Principal component 2 ({:.1f}%)'.format(100 * var_explained[1]))
        axs[1].set_ylabel('Principal component 2 ({:.1f}%)'.format(100 * var_explained[1]))
    else:
        axs[1].set_xlabel('Dimension 1')
        axs[0].set_ylabel('Dimension 2')
        axs[1].set_ylabel('Dimension 2')

    figure.config()
    axs[0].legend(bbox_to_anchor=(0.98, 1.0, 0.2, 0), loc='upper left')
    return figure


def evolution_plot(database, data, visit=None, metric='Volume', groups=None):
    metadata = database.get_metadata(which='all')
    metadata = metadata[['Patient', 'Group']]
    patients = database.get_patients()

    if groups is None:
        groups = sorted(list(metadata['Group'].dropna().unique()))

    figure = Figure(len(visit) - 1, sharex=True, sharey=True)
    axs = figure.get_axes()

    for v in range(1, len(visit)):
        df = pd.DataFrame([])
        patient_id = np.array([])
        patient_group = np.array([])
        for pat in patients:
            d = get_multiparametric_signature(pat, session=visit[v], parameters=[metric])
            d0 = get_multiparametric_signature(pat, session=visit[0], parameters=[metric, data])
            if (len(d) != 0) & (len(d0) != 0):
                d.insert(loc=0, column='V0', value=d0[metric])
                d.insert(loc=1, column='Parameter', value=d0[data])
                df = pd.concat((df, d)).reset_index(drop=True)
                patient_id = np.concatenate((patient_id, np.array([pat.id] * len(d))))
                patient_group = np.concatenate((patient_group, np.array(
                    [metadata.loc[metadata['Patient'] == pat.id, 'Group'].values[0]] * len(d))))

        ax = axs[(v - 1) // figure.nb_columns, (v - 1) % figure.nb_columns]
        for g in groups:
            dat = df[patient_group == g]
            ax.scatter(dat['Parameter'], dat[metric] / dat['V0'], s=5 * dat['V0'] / dat['V0'][dat['V0'] > 0].median(),
                       alpha=0.8, label=g)
        ax.set_ylim(max(0, ax.get_ylim()[0]), min(4, ax.get_ylim()[1]))
        ax.set_xlabel(data)
        if v == 1:
            ax.set_ylabel('Normalized volume')

    figure.config()
    return figure
