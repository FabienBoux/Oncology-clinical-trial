from clinlib.database import Database

from functions.config import *
from functions.graph import *
from functions.table import sample_size_table, probability_of_success

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

# fig = swimmer_plot(database, followup_time=12, followup_visits=['W6', 'M3', 'M6', 'M9', 'M12'], metric='Diameter')
fig = response_rate_plot(database, visits=['W6', 'M3', 'M6'], criteria='mRECIST', metric='Diameter')
# fig = volumetry_plot(database, visits=['W6', 'M3', 'M6', 'M9', 'M12'], stat='mean', metric='Diameter')

# fig = kaplan_meier_plot(database, followup_time=12, event='OS', adjust_ipfs=True, metric="Diameter",
#                         visits=config.get_value('visits'))

fig.save('test.png')
fig.close()

# probability_of_success(database, 400)
