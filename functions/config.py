import os
import configparser
import ast
import pandas as pd


class Config:
    def __init__(self):
        self.filename = 'config.ini'
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

    def get_value(self, key, section=None):
        if section is not None:
            return ast.literal_eval(self.config[section][key])
        else:
            for section in self.config.sections():
                if key in self.config[section]:
                    return ast.literal_eval(self.config[section][key])

    def set_value(self, val, key, section=None):
        if section is None:
            section = 'OTHER'
        if type(val) is list:
            val = str(val)

        if not self.is_section(section):
            self.config._sections[section] = {key: val}
        else:
            self.config[section][key] = val

    def extract_config_values(self, key):
        if key == 'list_metadata':
            metadata = pd.read_excel(self.get_value('metadata', section='PATH'))

            val = list(metadata.columns)
            val.remove('Patient')
            val.remove('Group')

            self.set_value(val, key='list_metadata', section='METADATA')

        return val
