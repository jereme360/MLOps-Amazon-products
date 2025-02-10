import pandas as pd
import json

class AmazonJSONParser:
    def parser(self, JSON_file):
        '''
        Parses Amazon product JSON file.
        '''
        df = pd.read_json(JSON_file, lines=True)
        return df