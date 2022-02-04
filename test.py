from clinlib.database import Database

from functions.config import *
from functions.graph import *

config = Config('config.ini')
config.read()

database = Database(config.get_value('database', section='PATH'), idlength=3)
database.add_resource({'metadata': os.path.join(config.get_value('database', section='PATH'),
                                                config.get_value('metadata', section='PATH'))})

kaplan_meier_plot(database)
