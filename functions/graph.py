from lifelines import CoxPHFitter
from matplotlib import pyplot as plt
from openpyxl import load_workbook
from scipy.stats import stats
from zepid.graphics import EffectMeasurePlot

from clinlib.displaying import Figure

import os
import datetime
import numpy as np
import pandas as pd

from collections import Counter


def swimmer_plot(database, followup_time=12, followup_visits=None):
    metadata = database.get_metadata(which='all')

    df = pd.DataFrame({'Patient': metadata['Patient'].values,
                       'Group': metadata['Group'].values,
                       'Start': metadata['Start'].values,
                       'End': metadata['End'].values,
                       }).dropna(how='all')

    if 'Expected' in df.columns:
        pd.concat((df, metadata['Expected']), axis=1)

    df['Event'] = ~np.isnat(df['End'])
    df['End'] = df['End'].fillna(datetime.datetime.now())

    df['Time'] = df['End']
    df['Time'] = (df['Time'] - df['Start']).dt.total_seconds() / 3600 / 24
    df['Time'] = df['Time'] / (365 / 12)

    df = df[~(df['Time'].isna())]

    # Plot
    figure = Figure(1)
    figure.set_figsize((1.5, len(df) / 20))
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
    pats = df[v].reset_index(drop=True)

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

    ev = df['Event'][v].values | (df['Time'][v].values > followup_time)

    for g in range(len(df['Group'].unique())):
        v_group = np.array([True if df['Group'][p] == df['Group'].unique()[g] else False for p in df[v].index])

        ax.barh(r2[v_group], y2[v_group], height=barWidth, color=colors[g], alpha=0.65,
                label='Group: {}'.format(df['Group'].unique()[g]))
        ax.plot(y2[v_group & ~ev], r2[v_group & ~ev], "_", color='black')
        ax.plot(y2[v_group & ~ev] + 0.1, r2[v_group & ~ev], ">", color='black')

        if 'Expected' in df.columns:
            ax.barh(r1, y1, height=barWidth / 2, color=['gray'] * len(y1), alpha=0.5,
                    label="Expected survival (if available)")

    # TODO: add responses
    # for p in range(len(r1)):
    #     t = None
    #     if pats.loc[p]['patient'] + '.xlsx' in os.listdir(database.folders['data']):
    #         dat = pd.read_excel(os.path.join(database.folders['data'], pats.loc[p]['patient'] + '.xlsx'),
    #                             sheet_name='RECIST')
    #         for i in ["W0_1h", "W0_4h", "Fr6"]:
    #             dat = dat[dat['Session'] != i]
    #         dat = dat.reset_index(drop=True)
    #
    #         previous_resp = ''
    #         for s in range(1, len(dat['Session'])):
    #             resp = dat['mRECIST response'][s]
    #             t = (1.5 if dat['Session'][s] == 'W6' else 3 if dat['Session'][s] == 'M3' else
    #             6 if dat['Session'][s] == 'M6' else 9 if dat['Session'][s] == 'M9' else
    #             12 if dat['Session'][s] == 'M12' else None)
    #
    #             if resp == 'PR':
    #                 ax.plot(t, r2[p], '.', color='green')
    #             if resp == 'CR':
    #                 ax.plot(t, r2[p], 'h', color='green')
    #             if resp == 'PD':
    #                 ax.plot(t, r2[p], '*', color='red')
    #
    #             if ((resp == 'PR') | (resp == 'CR')) & ((previous_resp == 'PR') | (previous_resp == 'CR')):
    #                 ax.plot([prev_t, t], [r2[p], r2[p]], color='green')
    #             if (resp == 'PD') & (previous_resp == 'PD'):
    #                 ax.plot([prev_t, t], [r2[p], r2[p]], color='red')
    #
    #             previous_resp = resp
    #             prev_t = t
    #
    #         if not t is None:
    #             if t == 1.5:
    #                 if (y2[p] - t) > 1.5:
    #                     ax.plot(t + 1.5, r2[p], color='black', marker='$?$')
    #             else:
    #                 if (y2[p] - t) > 3:
    #                     ax.plot(t + 3, r2[p], color='black', marker='$?$')
    #                 if (y2[p] > 12) & (t < 12):
    #                     ax.plot(12, r2[p], color='black', marker='$?$')
    #
    #     if (t is None) & (y2[p] >= 1.5):
    #         tt = [x for x in [1.5, 3, 6, 9, 12] if x <= y2[p]]
    #         ax.scatter(tt, [r2[p]] * len(tt), color='black', marker='$?$')

    ax.plot(np.nan, np.nan, ">", color='black', label='Censored patient')
    ax.plot(np.nan, np.nan, '.', color='green', label='Partial Response (PR)')
    ax.plot(np.nan, np.nan, 'h', color='green', label='Complete Response (CR)')
    ax.plot(np.nan, np.nan, color='green', label='Durable response')
    ax.plot(np.nan, np.nan, '*', color='red', label='Progressive Disease (PD)')
    ax.scatter(np.nan, np.nan, color='black', marker='$?$', label='Missing follow-up visit')
    ax.scatter(np.nan, np.nan, color='white',
               label='Absence of response or progression\ncorresponds to a stabilization')

    if followup_visits is not None:
        ax.set_xticks(
            [0] + [int(f[1:]) * (7 if f[0] == 'W' else (365 / 12) if f[0] == 'M' else 365 if f[0] == 'Y' else 1) for f
                   in followup_visits])
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


def forest_plot(database, list_metadata, model='lnHR', followup_time=12, group=None, n_min=5):
    metadata = database.get_metadata(which='all')

    df = pd.DataFrame({'Patient': metadata['Patient'].values,
                       'Group': metadata['Group'].values,
                       'Start': metadata['Start'].values,
                       'End': metadata['End'].values,
                       }).dropna(how='all')
    df = pd.concat((df, metadata[list_metadata]), axis=1)

    if group is None:
        group = df['Group'].unique()
        group = group[:2]

    df['Event'] = ~np.isnat(df['End'])
    df['End'] = df['End'].fillna(datetime.datetime.now())

    df['Time'] = df['End']
    df['Time'] = (df['Time'] - df['Start']).dt.total_seconds() / 3600 / 24
    df['Time'] = df['Time'] / (365 / 12)

    df = df[~(df['Time'].isna())]

    df['Time'].loc[df['Time'] > followup_time] = followup_time
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


def volumetry_plot(database, visits=None, which='targets', stat='mean', metric='Volume'):
    metadata = database.get_metadata()
    metadata = metadata[['Patient', 'Group']]

    group = list(metadata['Group'].unique())

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

            sessions = volume['Session'].unique()
            baseline = list(set([x if x.lower() == 'baseline' else x if float(x[1:]) == 0 else None for x in sessions]))
            baseline.remove(None)

            time = []
            evolution = pd.DataFrame([])
            ref = volume[volume['Session'] == baseline[0]]['Value'][lesion.values].sum()
            for ses in sessions:
                time.append(volume[volume['Session'] == ses]['Time'].mean() / (365 / 12))
                evolution.loc[0, ses] = 100 * (
                        volume[volume['Session'] == ses]['Value'][lesion.values].sum() - ref) / ref

            g = group.index(metadata[metadata['Patient'] == patients[p].id]['Group'].values[0])
            ax.plot(time, evolution.T, '-', color=colors[g], alpha=1 / (len(patients) ** .4))

            evolution.insert(loc=0, column='Group', value=group[g])
            df = df.append(evolution)

    if visits is None:
        visits = list(df.columns)
        visits.remove('Group')
    else:
        df = df[['Group'] + [i for i in df.columns if i in df.columns]]

    # TODO
    # stats.mannwhitneyu()

    for g in range(len(group)):
        x = np.array(
            [int(f[1:]) * (7 if f[0] == 'W' else (365 / 12) if f[0] == 'M' else 365 if f[0] == 'Y' else 1) for f in
             visits]) / (365 / 12)

        if stat == 'mean':
            y = df[df['Group'] == group[g]][visits].mean().values
            err = df[df['Group'] == group[g]][visits].std().values

            ax.plot(x, y, '--', color=colors[g], linewidth=4, label=group[g])
            ax.errorbar(x, y, err, fmt='none', color=colors[g], elinewidth=2, capsize=10, capthick=2)

            if g == 1:
                ax.errorbar(np.nan, np.nan, np.nan, fmt='none', color='k', elinewidth=2, capsize=10, capthick=2,
                            label="Standard deviations")

        else:
            y = df[df['Group'] == group[g]][visits].median().values
            err_low = y - df[df['Group'] == group[g]][visits].quantile(.25).values
            err_high = df[df['Group'] == group[g]][visits].quantile(.75).values - y

            ax.plot(x, y, '--', color=colors[g], linewidth=4, label=group[g])
            ax.errorbar(x, y, np.vstack([err_low, err_high]), fmt='none', color=colors[g], elinewidth=2, capsize=10,
                        capthick=2)
            if g == 1:
                ax.errorbar(np.nan, np.nan, np.nan, fmt='none', color='k', elinewidth=2, capsize=10, capthick=2,
                            label="25-75 quantiles")

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
    plt.ylabel('Changes in sum of the size of\nlesions compared to baseline (%)')
    plt.xlim([0, max(x) + 0.05])

    figure.config()
    return figure
