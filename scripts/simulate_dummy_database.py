import os
import datetime

import pandas as pd
import numpy as np


def generate_dummy_metadata(filename, nb_patients=200, nb_groups=2, ratio=.5, nb_meta=10):
    # TODO: use the ratio parameter
    metadata = pd.DataFrame({'Patient': ['%03d' % i for i in range(1, nb_patients + 1)],
                             'Group': np.random.randint(0, nb_groups, nb_patients),
                             'Start': datetime.date(2000, 1, 1) + np.array(
                                 [datetime.timedelta(np.random.randint(1, 365)) for i in range(nb_patients)]),
                             'End': datetime.date(2001, 1, 1) + np.array(
                                 [datetime.timedelta(np.random.randint(1, 365)) for i in range(nb_patients)])})

    for m in np.arange(nb_meta):

        if np.random.random() < 0.6:
            metadata.insert(loc=len(metadata.columns),
                            value=np.random.randint(0, 2, nb_patients),
                            column='Metadata {}'.format(m))
        elif np.random.random() < 0.6:
            metadata.insert(loc=len(metadata.columns),
                            value=np.random.randint(0, np.random.randint(3, 7), nb_patients),
                            column='Metadata {}'.format(m))
        else:
            metadata.insert(loc=len(metadata.columns),
                            value=np.random.random(nb_patients) * np.random.randint(0, 10, 1),
                            column='Metadata {}'.format(m))

        metadata.to_excel(filename, index=False)


def generate_dummy_data(metadata_file, data_folder=None, nb_param=1, sessions=np.arange(30, 365, 30),
                        dist_lesions=(9, 4)):
    if data_folder is None:
        data_folder = os.path.dirname(metadata_file)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    metadata = pd.read_excel(metadata_file)

    for pat in np.arange(len(metadata)):
        nb_lesions = 1 + abs(round(np.random.normal(dist_lesions[0], dist_lesions[1])))

        time = np.array([0] * nb_lesions)
        session = np.array(['M0'] * nb_lesions)
        lesion = np.array(['L{}'.format(i) for i in np.arange(nb_lesions)])

        data = dict()
        for par in np.arange(nb_param) + 1:
            data['Param {}'.format(par)] = pd.DataFrame(
                {'Time': time, 'Session': session, 'VOI': lesion, 'Value': np.random.random(nb_lesions)})

        # Initial volume and evolution according to a basic model and depending on group
        growth = abs(np.random.normal(2e-3, 1e-3))
        D_rt = 30
        alpha = 5e-1 / D_rt

        volume = abs(np.random.normal(10, 4, nb_lesions))
        v0 = volume.copy()
        if metadata['Group'][pat] == 0:
            D_eff = D_rt
        else:
            D_eff = (1 + np.array(
                [data[list(data.keys())[par - 1]]['Value'].values for par in np.arange(nb_param) + 1]).sum(
                axis=0)) * D_rt

        for ses in range(len(sessions)):
            time = np.append(time, np.array([sessions[ses]] * nb_lesions))
            session = np.append(session, np.array(['M{}'.format(ses + 1)] * nb_lesions))
            lesion = np.append(lesion, np.array(['L{}'.format(i) for i in np.arange(nb_lesions)]))
            # volume = np.append(volume, v0 * np.exp(growth * sessions[ses]) * np.exp(- alpha * D_eff))
            volume = np.append(volume, v0 * np.exp((1 - np.exp(- growth * sessions[ses]))) * np.exp(- alpha * D_eff))

        data['Volume'] = pd.DataFrame({'Time': time, 'Session': session, 'VOI': lesion, 'Value': volume})

        writer = pd.ExcelWriter(os.path.join(data_folder, '%03d.xlsx' % metadata['Patient'][pat]), engine='openpyxl')
        for par in data.keys():
            data[par].to_excel(writer, sheet_name=par, index=False)
        writer.save()


if __name__ == '__main__':
    # TODO: move this variable in config file
    folder = "C:\\Users\\Fabien Boux\\Code\\Oncology-clinical-trial\\data\\dummy"

    generate_dummy_metadata(os.path.join(folder, 'metadata.xlsx'))

    generate_dummy_data(metadata_file=os.path.join(folder, 'metadata.xlsx'),
                        data_folder=os.path.join(folder, 'data'))
