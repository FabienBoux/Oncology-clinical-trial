from matplotlib import pyplot as plt

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
    df['Time'] = df['Time'] / 30.5

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
            [0] + [int(f[1:]) * (7 if f[0] == 'W' else 30.4 if f[0] == 'M' else 365 if f[0] == 'Y' else 1) for f in
                   followup_visits])
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
