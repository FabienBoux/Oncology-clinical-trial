import numpy as np
import pandas as pd


def compute_sum(volume):
    volume.groupby(['Time', 'Value'])

    sessions = volume['Session'].unique()
    baseline = list(set([x if x.lower() == 'baseline' else x if float(x[1:]) == 0 else None for x in sessions]))
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
    baseline = list(set([x if x.lower() == 'baseline' else x if float(x[1:]) == 0 else None for x in sessions]))
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
            evolution.loc[0, ses] = 100 * (
                    volume[volume['Session'] == ses]['Value'].sum() - np.array(ref).min()) / np.array(ref).min()

    return time, evolution


def compute_mRECIST(volume, lesion):
    idx = [True if x in lesion[lesion == True].index else False for x in list(volume['VOI'].values)]
    time, evol_bl = compute_evolution(volume[idx], reference='baseline')
    time, evol_nd = compute_evolution(volume[idx], reference='nadir')
    _, vol = compute_sum(volume[idx])
    val = vol.copy().values[0]
    for i in range(1, len(vol.columns)):
        vol[vol.columns[i]] = val[i] - val[i-1]

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
