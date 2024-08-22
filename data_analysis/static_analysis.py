import numpy as np
import pandas as pd
import torch
import seaborn as sns

import matplotlib.pyplot as plt

from deepgo.data import get_data


class StaticAnalysisGO:
    def __init__(
            self,
    ):
        self.data_root = 'data/'
        self.output_root = '/output/'

    def read_data(
            self,
            data_root='data/',
            ont="mf",
            features_length=2560,
            features_column='esm2',
            test_data_file='test_data.pkl'
    ):
        terms_file = f'{data_root}/{ont}/terms.pkl'

        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}
        print('Terms', len(terms))

        ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
        iprs = ipr_df['interpros'].values
        iprs_dict = {v: k for k, v in enumerate(iprs)}

        if features_column == 'interpros':
            features_length = len(iprs_dict)

        train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
        valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
        test_df = pd.read_pickle(f'{data_root}/{ont}/{test_data_file}')

        train_data = get_data(train_df, iprs_dict, terms_dict, features_length, features_column)
        valid_data = get_data(valid_df, iprs_dict, terms_dict, features_length, features_column)
        test_data = get_data(test_df, iprs_dict, terms_dict, features_length, features_column)

        # datas = torch.concat([train_data[0], valid_data[0], test_data[0]], dim=0)
        labels = torch.concat([train_data[1], valid_data[1], test_data[1]], dim=0)
        self.show_data(labels.numpy())

    def show_data(self, data):
        protein_include_go = data.sum(axis=1)
        go_include_protein = data.sum(axis=0)
        flag = np.where(go_include_protein > 100, True, False)
        go_include_protein = go_include_protein[flag]


        fig = plt.figure(figsize=(10, 6))
        sns.distplot(protein_include_go)
        fig.savefig(f'{self.output_root}/protein_include_go.png')

        fig = plt.figure(figsize=(10, 6))
        sns.distplot(go_include_protein)
        fig.savefig(f'{self.output_root}/go_include_protein.png')


if __name__ == '__main__':
    static_analysis = StaticAnalysisGO()
    static_analysis.read_data()
