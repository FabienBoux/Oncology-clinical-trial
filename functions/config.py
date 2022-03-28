import os
import configparser
import ast
import pandas as pd


class Config:
    def __init__(self, filename='config.ini'):
        self.filename = filename
        self.pathfile = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], self.filename)

        self.config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})

    def set_filename(self, filename):
        self.filename = filename

    def exist(self):
        return os.path.isfile(self.pathfile)

    def read(self):
        if self.exist():
            self.config.read(self.pathfile, encoding='utf-8')

    def write(self):
        if self.exist():
            with open(self.pathfile, 'w') as configfile:
                self.config.write(configfile)

    def is_section(self, section):
        return section in self.config.sections()

    def is_key(self, key):
        for section in self.config.sections():
            if key in self.config[section]:
                return True
        return False

    def get_value(self, key, section=None):
        if section is not None:
            val = self.config[section][key]
            return ast.literal_eval(val) if (val.startswith('[') & val.endswith(']')) else val
        else:
            for section in self.config.sections():
                if key in self.config[section]:
                    val = self.config[section][key]
                    return ast.literal_eval(val) if (val.startswith('[') & val.endswith(']')) else val

    def remove_value(self, key, section=None):
        if section is not None:
            if key in self.config[section]:
                self.config.remove_option(section, key)
        else:
            for section in self.config.sections():
                if key in self.config[section]:
                    return self.config.remove_option(section, key)

    def remove_section(self, section):
        if section is not None:
            return self.config.remove_section(section)

    def set_value(self, val, key, section=None):
        if section is None:
            section = 'OTHER'
        if type(val) is list:
            val = str(val)

        if not self.is_section(section):
            self.config.add_section(section)
        self.config.set(section, key, val)

    def extract_config_values(self, key):
        if key == 'list_metadata':
            metadata = pd.read_excel(
                os.path.join(self.get_value('database', section='PATH'), self.get_value('metadata', section='PATH')))

            val = list(metadata.columns)
            val.remove('Patient')
            val.remove('Group')

            self.set_value(val, key='list_metadata', section='METADATA')

        if key == 'list_data':
            parameters = list()
            for filename in os.listdir(os.path.join(self.get_value('database', section='PATH'), 'data')):
                xls = pd.ExcelFile(os.path.join(self.get_value('database', section='PATH'), 'data', filename))
                parameters = parameters + xls.sheet_names

                val = list(set(parameters))

            self.set_value(val, key='list_data', section='DATA')

        if key == 'followup':
            # TODO: correct next code
            # metadata = pd.read_excel(self.get_value('followup', section='OTHER'))
            #
            # val = list(metadata.columns)
            # val.remove('Patient')
            # val.remove('Group')
            #
            # self.set_value(val, key='followup', section='OTHER')
            pass

        return val
