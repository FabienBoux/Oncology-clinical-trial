import datetime

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
from zepid.graphics import EffectMeasurePlot

from clinlib.displaying import Figure
from functions.utils import compute_evolution, compute_revised_RECIST, visit_to_time


def swimmer_plot(database, followup_time=None, followup_visits=None, metric='Volume'):
    metadata = database.get_metadata(which='all')

    df = pd.DataFrame({'Patient': metadata['Patient'].values,
                       'Group': metadata['Group'].values,
                       'Start': metadata['Start'].values,
                       'End': metadata['End'].values,
                       'Event': metadata['Event'].values,
                       }).dropna(how='all')

    if 'Expected' in df.columns:
        pd.concat((df, metadata['Expected']), axis=1)

    df['End'] = df['End'].fillna(datetime.datetime.now())

    df['Time'] = df['End'].copy()
    df['Time'] = (df['Time'] - df['Start']).dt.total_seconds() / 3600 / 24
    df['Time'] = df['Time'] / (365 / 12)

    df = df[~(df['Time'].isna())]

    if followup_time is None:
        followup_time = df['Time'].max()
    else:
        df.loc[df['Time'] > followup_time, 'Time'] = followup_time

    # Plot
    figure = Figure(1)
    figure.set_figsize((1, len(df) / 30))
    ax = figure.get_axes()[0, 0]
    colors = figure.get_colors()

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

    # TODO: add cause of death if available
    # jumps = list(Counter(df['cause']).values())
    # jumps.reverse()
    # jumps = np.array(jumps).cumsum()
    # for i in range(len(jumps) - 1):
    #     r1[-(jumps[i]):] = r1[-(jumps[i]):] + 1
    #     r2[-(jumps[i]):] = r2[-(jumps[i]):] + 1

    censored = df['Event'][v].isna()

    for g in range(len(df['Group'].unique())):
        v_group = np.array([True if df['Group'][p] == df['Group'].unique()[g] else False for p in df[v].index])

        ax.barh(r2[v_group], y2[v_group], height=barWidth, color=colors[g], alpha=0.65,
                label='Group: {}'.format(df['Group'].unique()[g]))
        ax.plot(y2[v_group & censored], r2[v_group & censored], "_", color='black')
        ax.plot(y2[v_group & censored] + followup_time * .01, r2[v_group & censored], ">", color='black')

        if 'Expected' in df.columns:
            ax.barh(r1, y1, height=barWidth / 2, color=['gray'] * len(y1), alpha=0.5,
                    label="Expected survival (if available)")

    patients = database.get_patients(patient_ids)
    for p in range(len(r1)):
        volume = patients[p].get_data(metric)
        lesion = patients[p].get_lesion()

        flwt = followup_visits.copy()
        if (not lesion.empty) & (not volume.empty):
            _, response = compute_revised_RECIST(volume, lesion)

            previous_resp = ''
            for s in response.columns[1:]:
                resp = response[s].values[0]
                t = visit_to_time(s) / (365 / 12)
                if flwt is not None:
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
        ax.set_xticklabels(['0'] + followup_visits)

    ax.set_yticks(r2)
    ax.set_yticklabels(list(df['Patient'][v].values))

    ax.set_xlabel('Time (months) / Follow-up visit')
    ax.set_ylabel('Patients (n = {})'.format(v.sum()))
    ax.set_xlim([ax.get_xlim()[0], followup_time + 0.2])
    ax.set_ylim([r2[0] - barWidth, r2[-1] + barWidth])

    figure.config()
    ax.grid(True, which='major', axis='x', linestyle='--')
    ax.grid(False, axis='y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.7))

    return figure


def forest_plot(database, list_metadata, model='lnHR', followup_time=None, group=None, n_min=5):
    metadata = database.get_metadata(which='all')

    df = pd.DataFrame({'Patient': metadata['Patient'].values,
                       'Group': metadata['Group'].values,
                       'Start': metadata['Start'].values,
                       'End': metadata['End'].values,
                       'Event': metadata['Event'].values,
                       }).dropna(how='all')
    df = pd.concat((df, metadata[list_metadata]), axis=1)

    if group is None:
        group = df['Group'].unique()
        group = group[:2]

    df['End'] = df['End'].fillna(datetime.datetime.now())
    df['Event'] = df['Event'].fillna(0)

    df['Time'] = (df['End'] - df['Start']).dt.total_seconds() / 3600 / 24
    df['Time'] = df['Time'] / (365 / 12)

    df = df[~(df['Time'].isna())]

    if followup_time is not None:
        df.loc[df['Time'] > followup_time, 'Time'] = followup_time

    df = df.replace(group[0], 0).replace(group[1], 1)

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
                cph = CoxPHFitter()
                cph.fit(df[['Group', 'Time', 'Event']][v], duration_col='Time', event_col='Event')

                measure.append(round(cph.params_[0], 2))
                lower.append(round(cph.confidence_intervals_.values[0, 0], 2))
                upper.append(round(cph.confidence_intervals_.values[0, 1], 2))

                n_rt.append((df[v]['Group'] == 0).sum())
                n_ax.append((df[v]['Group'] == 1).sum())
                labs.append("{}: {}   (n={}/{})".format(list_metadata[i], str(j), (df[v]['Group'] == 0).sum(),
                                                        (df[v]['Group'] == 1).sum()))

    cph = CoxPHFitter()
    cph.fit(df[['Group', 'Time', 'Event']], duration_col='Time', event_col='Event')

    measure.append(round(cph.params_[0], 2))
    lower.append(round(cph.confidence_intervals_.values[0, 0], 2))
    upper.append(round(cph.confidence_intervals_.values[0, 1], 2))
    labs.append("Overall (n={}/{})".format((df['Group'] == 0).sum(), (df['Group'] == 1).sum()))
    n_rt.append((df['Group'] == 0).sum())
    n_ax.append((df['Group'] == 1).sum())

    if model == 'HR':
        measure = np.exp(measure)
        lower = np.exp(lower)
        upper = np.exp(upper)

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(effectmeasure=model, center=(0 if model == "lnHR" else 1))
    p.colors(pointshape="D", pointcolor=colors[0], errorbarcolor=colors[0])

    x_min = round(max([1.05, (max(upper) * 1.05 if max(upper) > 0 else max(upper) * 0.95)]), 2)
    x_max = round(min([0.95, (min(lower) * 0.95 if min(lower) > 0 else min(lower) * 1.05)]), 2)
    ax = p.plot(figsize=(7, 3), t_adjuster=0.09, max_value=x_min, min_value=x_max)

    plt.suptitle("Subgroup", x=-0.1, y=0.98)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

    return 1


def volumetry_plot(database, visits=None, which='targets', stat='mean', metric='Volume', groups=None):
    metadata = database.get_metadata()
    metadata = metadata[['Patient', 'Group']]

    if groups is None:
        groups = sorted(list(metadata['Group'].unique()))

    figure = Figure(1)
    figure.set_figsize((1.3, 1))
    ax = figure.get_axes()[0, 0]
    colors = figure.get_colors()

    df = pd.DataFrame([])

    patients = database.get_patients()
    for p in range(len(patients)):
        volume = patients[p].get_data(metric)
        lesion = patients[p].get_lesion()

        if (not lesion.empty) & (not volume.empty):
            idx = [True if x in lesion[lesion].index else False for x in list(volume['VOI'].values)]
            time, evolution = compute_evolution(volume[idx])

            g = groups.index(metadata[metadata['Patient'] == patients[p].id]['Group'].values[0])
            ax.plot(time, evolution.values[0], '-', color=colors[g], alpha=1 / (len(patients) ** .4))

            evolution.insert(loc=0, column='Group', value=groups[g])
            df = pd.concat((df, evolution))

    if visits is None:
        visits = list(df.columns)
        visits.remove('Group')
    else:
        df = df[['Group'] + [i for i in df.columns if i in df.columns]]

    # TODO
    # stats.mannwhitneyu()

    for g in range(len(groups)):
        x = np.array(
            [int(f[1:]) * (7 if f[0] == 'W' else (365 / 12) if f[0] == 'M' else 365 if f[0] == 'Y' else 1) for f in
             visits]) / (365 / 12)

        if stat == 'mean':
            y = df[df['Group'] == groups[g]][visits].mean().values
            err = df[df['Group'] == groups[g]][visits].std().values

            ax.plot(x, y, '--', color=colors[g], linewidth=4, label="Mean group {}".format(groups[g]))
            ax.errorbar(x, y, err, fmt='none', color=colors[g], elinewidth=2, capsize=10, capthick=2)

            if g == 1:
                ax.errorbar(np.nan, np.nan, np.nan, fmt='none', color='k', elinewidth=2, capsize=10, capthick=2,
                            label="Standard deviations")

        else:
            y = df[df['Group'] == groups[g]][visits].median().values
            err_low = y - df[df['Group'] == groups[g]][visits].quantile(.2).values
            err_high = df[df['Group'] == groups[g]][visits].quantile(.8).values - y

            ax.plot(x, y, '--', color=colors[g], linewidth=4, label="Median group {}".format(groups[g]))
            ax.errorbar(x, y, np.vstack([err_low, err_high]), fmt='none', color=colors[g], elinewidth=2, capsize=10,
                        capthick=2)
            if g == 1:
                ax.errorbar(np.nan, np.nan, np.nan, fmt='none', color='k', elinewidth=2, capsize=10, capthick=2,
                            label="20-80 quantiles")

    # for t in range(1, len(tt)):
    #     if stat == 'mean':
    #         plt.text(x_ax[t] - 0.5, plt.ylim()[1] * 1.45, '  {:.1f}%'.format(100 * (y_rt[t] - y_ax[t]) / y_rt[t]))
    #     else:
    #         plt.text(x_ax[t] - 0.5, plt.ylim()[1] * 1.45, '  {:.1f}%'.format(100 * (z_rt[t] - z_ax[t]) / z_rt[t]))
    #     if (n_rt[t] > 3) & (n_ax[t] > 3):
    #         plt.text(x_ax[t] - 0.5, plt.ylim()[1] * 1.25, ' p = {:.2f}'.format(pval[t]))
    #     plt.text(x_ax[t] - 0.5, plt.ylim()[1] * 1.05, 'n = {}$^*$/ {}$^°$'.format(int(n_rt[t]), int(n_ax[t])))

    ax.set_xticks(x)
    ax.set_xticklabels(visits)

    plt.xlabel('Times (months)')
    plt.ylabel('Changes in sum of the size of lesions\ncompared to baseline (%)')
    plt.xlim([0, max(x) + 0.05])

    figure.config()
    return figure


def kaplan_meier_plot(database, event='OS', followup_time=None, cutoff_date=None, groups=None, metric='Volume',
                      adjust_ipfs=False, visits=None):
    cutoff_date = (datetime.datetime.now() if cutoff_date is None
                   else datetime.datetime.strptime(cutoff_date, '%d/%m/%y'))

    metadata = database.get_metadata(which='all')
    df = metadata[['Patient', 'Group', 'Start', 'End', 'Event']].dropna(how='all')

    if groups is None:
        groups = sorted(list(df['Group'].unique()))

    df['End'] = df['End'].fillna(datetime.datetime.now())

    df['Time'] = (df['End'] - df['Start']).dt.total_seconds() / 3600 / 24
    df['Time'] = df['Time'] / (365 / 12)

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
            lesion = patient.get_lesion()

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

    ax.plot([0, followup_time], [0.5, 0.5], '--', color="black", alpha=0.9,
            label=('Median PFS time (mPFST)' if event == 'PFS' else 'Median survival time (mST)'))
    ax.plot([np.nan], [np.nan], '.', color='k', label='Censored patients')

    # ax.plot(np.nan, np.nan, '-', color='k',
    #         label='Log-rank test: p={:.2f} (n={})\nwith censored: p={:.2f} (n={})'.format(results.p_value, v.sum(),
    #                                                                                       results_all.p_value,
    #                                                                                       len(rt)))
    ax.plot(np.nan, np.nan, '-', color='k',
            label='Log-rank test: p={:.2f} (n={})'.format(results_all.p_value, len(v)))

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

    figure.config()
    return figure


def response_rate_plot(database, visits=None, criteria='rRECIST', cutoff_date=None, metric='Volume', groups=None):
    cutoff_date = (datetime.datetime.now() if cutoff_date is None
                   else datetime.datetime.strptime(cutoff_date, '%d/%m/%y'))

    metadata = database.get_metadata(which='all')

    if groups is None:
        groups = sorted(list(metadata['Group'].unique()))

    df = pd.DataFrame([])
    patients = database.get_patients()
    for p in range(len(patients)):
        volume = patients[p].get_data(metric)
        lesion = patients[p].get_lesion()

        if (not lesion.empty) & (not volume.empty):
            if criteria == 'rRECIST':
                _, response = compute_revised_RECIST(volume, lesion)
            elif criteria == 'mRECIST':
                _, response = compute_revised_RECIST(volume, patients[p].get_lesion(max_number=np.inf))
            else:
                _, response = compute_revised_RECIST(volume, lesion)

            g = groups.index(metadata[metadata['Patient'] == patients[p].id]['Group'].values[0])
            response.insert(loc=0, column='Group', value=groups[g])

            df = pd.concat((df, response))

    if visits is None:
        visits = list(df.columns)
        visits.remove('Group')
    else:
        df = df[['Group'] + [i for i in df.columns if i in df.columns]]

    figure = Figure(1)
    figure.set_figsize((1.3, 1))
    ax = figure.get_axes()[0, 0]
    colors = figure.get_colors()

    labels = list(df.columns)[2:]
    width = .8
    x = np.arange(len(labels))

    for g in range(len(groups)):
        ax.bar(x - width / 2 + g * width / len(groups),
               ((df[labels][df['Group'] == groups[g]] == 'PR') | (
                       df[labels][df['Group'] == groups[g]] == 'CR')).sum().values,
               width / len(groups), color=colors[g], label='Group: ' + groups[g])
        ax.bar(x - width / 2 + g * width / len(groups),
               (df[labels][df['Group'] == groups[g]] == 'CR').sum().values,
               width / len(groups), color=colors[g], edgecolor='black', hatch='//')

    ax.bar(np.nan, np.nan, width, color='white', edgecolor='black', hatch='//', label='CR')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel('Sessions')
    ax.set_ylabel('Number of responses (PR or CR)')

    figure.config()
    return figure
