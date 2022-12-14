from .base import AbstractDataset

import pandas as pd

from datetime import date


class OULADDataset(AbstractDataset):
    def __init__(self, args):
        super().__init__(args)
        self.drop_dup = args.drop_dup
    @classmethod
    def code(cls):
        return 'oulad'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def url(cls):
        return  'https://analyse.kmi.open.ac.uk/open_dataset/download'

    @classmethod
    def all_raw_file_names(cls):
        return ['assessments.csv',
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
       'rating']
        if self.drop_dup:
            # whether to drop duplicates at same timestamp
            df = df.drop_duplicates(subset=['uid', 'sid', 'timestamp'])
        return df


