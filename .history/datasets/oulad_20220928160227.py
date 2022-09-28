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

    # @classmethod
    # def all_raw_file_names(cls):
    #     return ['README',
    #             'movies.dat',
    #             'ratings.dat',
    #             'users.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('OULAD/studentVle.csv')
        df = pd.read_csv(file_path)
        return df


