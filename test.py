from clinlib.database import Database

from functions.config import *
from functions.graph import *
from functions.table import sample_size_table, probability_of_success, correlation_table
from functions.utils import power_probability

config = Config('real.ini')
config.read()

database = Database(config.get_value('database', section='PATH'), idlength=6)
database.add_resource({'metadata': os.path.join(config.get_value('database', section='PATH'),
                                                config.get_value('metadata', section='PATH'))})

metadata = database.get_metadata(which='all')

group = metadata['Group']
group_labels = group.unique()

list_selected = ['Age', 'Cancer type', 'ECOG PS', 'Gender', 'Number of metastases',
                 'Presence of extracranial metastases at inclusion']

# fig = dim_reduction_plot(database, list_data=['ADC', 'FLAIR', 'SWI', 'T1'], visit='Baseline', method='PCA',
#                          outliers=False)

# probability_of_success_plot(database, 100, event='PFS', metric='Diameter')
#
# probability_of_success(database, 100, event='PFS', metric='Diameter')
#
# test = power_probability(pd.DataFrame([]), 100, alpha=.05, condition='PPoS', ratio=1)
#
# t1 = sample_size_table(database, followup_time=12, group=None, criteria='HR', event='OS',
#                   metric='Diameter', visits=None, adjust_ipfs=True)
# t2 = sample_size_table(database, followup_time=12, group=None, criteria='HR', event='PFS',
#                   metric='Diameter', visits=None, adjust_ipfs=True)

# fig = forest_plot(database, list_selected, model='TR', followup_time=None, groups=['WBRT', 'AGuIX'], n_min=5)
# fig = swimmer_plot(database, followup_time=12, followup_visits=['W6', 'M3', 'M6', 'M9', 'M12'], metric='Diameter', groups =['WBRT','AGuIX'])
fig = response_rate_plot(database, visits=['W6', 'M3', 'M6'], criteria='mRECIST', metric='Diameter')
# fig = volumetry_plot(database, visits=['W6', 'M3', 'M6', 'M9', 'M12'], stat='mean', metric='Diameter')

fig = evolution_plot(database, 'SE', visit=['Baseline', 'W6', 'M3', 'M6'], metric='Diameter')

# fig = correlation_matrix_plot(database, list_data=['ADC', 'FLAIR', 'SWI', 'T1'], visit='Baseline', threshold=True)

# fig = kaplan_meier_plot(database, followup_time=12, event='OS', adjust_ipfs=True, metric="Diameter",
#                         visits=config.get_value('visits'))

fig.save('test.png')
fig.close()

# probability_of_success(database, 400)
