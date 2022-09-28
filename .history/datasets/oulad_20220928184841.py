from .base import AbstractDataset

import pandas as pd

from datetime import date


class OULADDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'oulad'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['assessment.csv',
                'courses.csv',
                'studentAssessment.csv',
                'studentInfo.csv',
                'studentRegistration.csv',
                'studentVle.csv',
                'vle.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('studentVle.csv')
        df = pd.read_csv(file_path)
        df.columns = ['code_module', 'code_presentation', 'uid', 'sid', 'timestamp',
       'ratings']
        return df


