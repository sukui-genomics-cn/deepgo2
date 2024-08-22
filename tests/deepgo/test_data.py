import unittest


class TestData(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.output_dir = "../../output"
        pass

    def test_read_pkl(self):
        # test_data.TestData.test_read_pkl
        import pickle
        import pandas as pd

        pkl_file = '/home/share/huadjyin/home/s_sukui/02_data/08_protein/GO/deepgo2/data/cc/test_data.pkl'

        # read pkl file
        with open(pkl_file, 'rb') as f:
            df_test_data = pd.read_pickle(pkl_file)
        df_head = df_test_data.head()
        df_head.to_csv(f'{self.output_dir}/test_data_head.csv', index=False)
        print(df_test_data.keys())

    def test_pyg(self):
        # test_data.TestData.test_pyg
        pass
