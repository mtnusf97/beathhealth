import argparse
import copy
import os
import pickle
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X) or not torch.is_tensor(y):
            X = torch.from_numpy(X)
            y = torch.from_numpy(y)

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MLP(nn.Module):
    """
        Multilayer Perceptron for regression.
    """

    def __init__(self, n_in, n_out):
        super().__init__()
        #         self.layers = nn.Sequential(
        #             nn.Linear(n_in, 64),
        #             nn.ReLU(),
        #             nn.Linear(64, 32),
        #             nn.ReLU(),
        #             nn.Linear(32, n_out)
        #         )

        #         self.layers = nn.Sequential(
        #             nn.Linear(n_in, 32),
        #             nn.ReLU(),
        #             nn.Linear(32, 64),
        #             nn.ReLU(),
        #             nn.Linear(64, 64),
        #             nn.ReLU(),
        #             nn.Linear(64, 32),
        #             nn.ReLU(),
        #             nn.Linear(32, 16),
        #             nn.ReLU(),
        #             nn.Linear(16, n_out)
        #         )

        self.layers = nn.Sequential(
            nn.Linear(n_in, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_out)
        )

    #         self.layers = nn.Sequential(
    #             nn.Linear(n_in, 32),
    #             nn.ReLU(),
    #             nn.Linear(32, 32),
    #             nn.ReLU(),
    #             nn.Linear(32, 32),
    #             nn.ReLU(),
    #             nn.Linear(32, n_out)
    #         )

    def forward(self, x):
        """
          Forward pass
        """
        return self.layers(x)


def run_linear_regression(X, y, test_size):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
    model = LinearRegression().fit(X_train, y_train)
    X_output = model.predict(X_val)
    loss_val = np.dot(X_output - y_val, X_output - y_val) / len(X_val)
    corr_val = np.corrcoef(X_output, y_val)[0, 1]
    r2_val = model.score(X_val, y_val)

    best_val_results = dict(loss=loss_val, corr=corr_val, r2=r2_val)

    return best_val_results, model


if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--AllResSaveFolder", help="all results save folder")
    parser.add_argument("-d", "--DataframeSavePath", help="dataframe save path")
    parser.add_argument("-n", "--NumOfShuffles", help="dataframe save path")
    parser.add_argument("-m", "--ModelPath", help="path to save model")
    args = parser.parse_args()
    dataframe_savepath = args.DataframeSavePath
    all_res_savefolder = args.AllResSaveFolder
    model_path = args.ModelPath
    n_shuff = int(args.NumOfShuffles)

    # data
    df_to_run = pd.read_csv('../data/deep_factor_test_retest_imputed_all_standardized.csv')
    data_arr = df_to_run.iloc[:, 2:].to_numpy()
    X = data_arr[:, :12]
    y_total = data_arr[:, 12:]

    # running
    results_df = dict()
    results_df['name'] = ['loss_null', 'loss_alternative', 'loss_pvalue', 'is_loss_sig',
                          'corr_null', 'corr_alternative', 'corr_pvlue', 'is_corr_sig',
                          'r2_null', 'r2_alternative', 'r2_pvalue', 'is_r2_sig']
    for idx in range(5):
        y = copy.deepcopy(y_total[:, idx])
        alternative_res = dict(loss=list(), corr=list(), r2=list())
        best_model = None
        best_corr = -np.inf
        for i in tqdm(range(n_shuff)):
            # np.random.shuffle(y)
            res, model = run_linear_regression(X, y, 0.2)
            alternative_res['loss'].append(res['loss'])
            alternative_res['corr'].append(res['corr'])
            alternative_res['r2'].append(res['r2'])
            if res['corr'] > best_corr:
                best_corr = copy.deepcopy(res['corr'])
                best_model = copy.deepcopy(model)

        with open(model_path + '_' + str(idx) + '.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        null_res = dict(loss=list(), corr=list(), r2=list())
        for i in tqdm(range(n_shuff)):
            np.random.shuffle(y)
            shuf_res, _ = run_linear_regression(X, y, 0.2)
            null_res['loss'].append(shuf_res['loss'])
            null_res['corr'].append(shuf_res['corr'])
            null_res['r2'].append(shuf_res['r2'])

        all_res = dict(alternative=alternative_res, null=null_res)

        col_name = df_to_run.columns[idx + 14]
        with open(os.path.join(all_res_savefolder, f"{str(idx)}_{col_name}.pkl"), 'wb') as f:
            pickle.dump(all_res, f)

        p_loss = stats.ttest_ind(alternative_res['loss'], null_res['loss'], alternative='less').pvalue
        p_corr = stats.ttest_ind(alternative_res['corr'], null_res['corr'], alternative='greater').pvalue
        p_r2 = stats.ttest_ind(alternative_res['r2'], null_res['r2'], alternative='greater').pvalue

        sig_loss = 'Yes' if p_loss <= 0.05 else 'No'
        sig_corr = 'Yes' if p_corr <= 0.05 else 'No'
        sig_r2 = 'Yes' if p_r2 <= 0.05 else 'No'

        # results_df[col_name] = [int(idx),
        #                         round(np.mean(null_res['loss']), 4), round(np.mean(alternative_res['loss']), 4),
        #                         round(p_loss, 3), sig_loss,
        #                         round(np.mean(null_res['corr']), 4), round(np.mean(alternative_res['corr']), 4),
        #                         round(p_corr, 3), sig_corr,
        #                         round(np.mean(null_res['r2']), 4), round(np.mean(alternative_res['r2']), 4),
        #                         round(p_r2, 3), sig_r2]
        results_df[col_name] = [round(np.mean(null_res['loss']), 4), round(np.mean(alternative_res['loss']), 4),
                                '%.2E' % Decimal(p_loss), sig_loss,
                                round(np.mean(null_res['corr']), 4), round(np.mean(alternative_res['corr']), 4),
                                '%.2E' % Decimal(p_corr), sig_corr,
                                round(np.mean(null_res['r2']), 4), round(np.mean(alternative_res['r2']), 4),
                                '%.2E' % Decimal(p_r2), sig_r2]

        print(f'idx {idx} \np_value of loss: {"%.2E" % Decimal(p_loss)} \n'
              f'p_value of corr: {"%.2E" % Decimal(p_corr)} \np_value of r2: {"%.2E" % Decimal(p_r2)}')
        print('\n================\n')

    pd.DataFrame(results_df).to_csv(dataframe_savepath, index_label=True)
