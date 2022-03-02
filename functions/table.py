import datetime

import numpy as np
import pandas as pd

from functions.utils import sample_size_calculation, predictive_probability, compute_revised_RECIST, power_probability


def sample_size_table(database, followup_time=None, group=None, criteria='HR', event='OS', metric='Volume', visits=None,
                      adjust_ipfs=True):
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

    df['End'] = df['End'].fillna(datetime.datetime.now())
    df['Event'] = df['Event'].fillna(0)

    df['Time'] = (df['End'] - df['Start']).dt.total_seconds() / 3600 / 24
    df['Time'] = df['Time'] / (365 / 12)

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

    alphas = [0.025, 0.05, 0.1]
    powers = [0.9, 0.85, 0.8]
    tab = [[sample_size_calculation(df, alpha=a, power=p, ratio=1, criteria=criteria) for a in alphas] for p in powers]

    return pd.DataFrame(data=tab, columns=alphas, index=powers)


def probability_of_success(database, n_total, followup_time=None, group=None, event='OS', metric='Volume', visits=None,
                           adjust_ipfs=True):
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

    df['Time'] = (df['End'] - df['Start']).dt.total_seconds() / 3600 / 24
    df['Time'] = df['Time'] / (365 / 12)

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

    dat = [round(power_probability(df, n_total, condition='PoS') * 100, 1),
           round(power_probability(df, n_total, condition='CP') * 100, 1),
           round(power_probability(df, n_total, condition='PPoS') * 100, 1)]

    dat.append("{}-{}".format(round(np.array(dat).min()), round(np.array(dat[1:]).max())))

    # return predictive_probability(df, n_total)
    return pd.DataFrame(data=dat, columns=['Probabilities (%)'], index=['PoS', 'CP', 'PPoS', 'Overall'])
