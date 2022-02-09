from clinlib.database import Database

from functions.config import *
from functions.graph import *

config = Config('config.ini')
config.read()

database = Database(config.get_value('database', section='PATH'), idlength=3)
database.add_resource({'metadata': os.path.join(config.get_value('database', section='PATH'),
                                                config.get_value('metadata', section='PATH'))})

fig = kaplan_meier_plot(database, followup_time=12, event='PFS', adjust_ipfs=False,
                        visits=['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12'])
fig.save('test.png')
fig.close()
