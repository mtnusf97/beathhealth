import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import os
import time
from sklearn.metrics import r2_score
import warnings
import pickle
import argparse
from scipy import stats
from decimal import Decimal

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


def run_deep(X, y, idx, test_size,
             train_batch_size, val_batch_size,
             learning_rate,
             tensorboard_path,
             print_iteration=1,
             n_epochs=100,
             tensorboard=True,
             do_print=True):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    # Prepare dataset
    train_dataset = Dataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = Dataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    # Initialize the MLP
    mlp = MLP(n_in=X_train.shape[1], n_out=1)

    # Define the loss function and optimizer
    loss_function = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    if tensorboard:
        writer = SummaryWriter(tensorboard_path)

    # Run the training loop
    total_iteration = 0
    best_val_results = dict(loss=np.inf, corr=-np.inf, r2=-np.inf)
    best_val_mlp = None
    for epoch in range(n_epochs):
        outputs_train_arr = np.array([])
        targets_train_arr = np.array([])

        current_loss = 0.0
        n_train_data = 0
        for inputs, targets in train_loader:
            total_iteration += 1
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            n_train_data += inputs.shape[0]

            optimizer.zero_grad()

            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

            outputs_detach = torch.detach(outputs)
            targets_detach = torch.detach(targets)
            outputs_train_arr = np.append(outputs_train_arr, torch.Tensor.numpy(outputs_detach).reshape(1, -1)[0])
            targets_train_arr = np.append(targets_train_arr, torch.Tensor.numpy(targets_detach).reshape(1, -1)[0])

            if total_iteration % print_iteration == 0:
                loss_val = 0
                n_val_data = 0
                outputs_val_arr = np.array([])
                targets_val_arr = np.array([])

                for inputs_val, targets_val in val_loader:
                    inputs_val, targets_val = inputs_val.float(), targets_val.float()
                    targets_val = targets_val.reshape((targets_val.shape[0], 1))
                    n_val_data += inputs_val.shape[0]
                    outputs_val = mlp(inputs_val)
                    loss_val += loss_function(outputs_val, targets_val).item()
                    outputs_val_detach = torch.detach(outputs_val)
                    targets_val_detach = torch.detach(targets_val)
                    outputs_val_arr = np.append(outputs_val_arr,
                                                torch.Tensor.numpy(outputs_val_detach).reshape(1, -1)[0])
                    targets_val_arr = np.append(targets_val_arr,
                                                torch.Tensor.numpy(targets_val_detach).reshape(1, -1)[0])

                corrcoef = np.corrcoef(targets_val_arr, outputs_val_arr)[0, 1]
                r_squared = r2_score(targets_val_arr, outputs_val_arr)
                train_loss = current_loss / n_train_data
                val_loss = loss_val / n_val_data

                if tensorboard:
                    writer.add_scalars(f'{str(idx)}_loss', {
                        'train': train_loss,
                        'validation': val_loss,
                    }, total_iteration)
                    writer.add_scalars(f'{str(idx)}_corr-R', {
                        'corr': corrcoef,
                        'r': r_squared,
                    }, total_iteration)

                if val_loss < best_val_results['loss']:
                    best_val_results['loss'] = copy.deepcopy(val_loss)
                    best_val_mlp = copy.deepcopy(mlp)

                best_val_results['corr'] = copy.deepcopy(corrcoef) if corrcoef > best_val_results['corr'] \
                    else best_val_results['corr']
                best_val_results['r2'] = copy.deepcopy(r_squared) if r_squared > best_val_results['r2'] \
                    else best_val_results['r2']

                current_loss = 0.0
                n_train_data = 0

    print_str = (f"best of {str(idx).rjust(3)} => loss: {str(round(best_val_results['loss'], 2)).rjust(11)}, " +
                 f"corr: {str(round(best_val_results['corr'], 2)).rjust(5)}, " +
                 f"r2: {str(round(best_val_results['r2'], 2)).rjust(5)}")
    if do_print:
        print(print_str)

    if tensorboard:
        writer.add_text(str(idx), print_str)
        del writer

    return print_str, best_val_results, best_val_mlp


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
        best_mlp = None
        best_corr = -np.inf
        for i in tqdm(range(n_shuff)):
            # np.random.shuffle(y)
            tensorboard_path = os.path.join('tensorboard5/', str(idx) + '_' + str(int(round(time.time() % 1, 5) * 1e5)))
            _, res, mlp = run_deep(X, y, idx, 0.2,
                                   train_batch_size=10, val_batch_size=5,
                                   learning_rate=1e-3,
                                   tensorboard_path=tensorboard_path,
                                   print_iteration=1,
                                   n_epochs=100,
                                   tensorboard=False,
                                   do_print=False)
            alternative_res['loss'].append(res['loss'])
            alternative_res['corr'].append(res['corr'])
            alternative_res['r2'].append(res['r2'])
            if res['corr'] > best_corr:
                best_corr = copy.deepcopy(res['corr'])
                best_mlp = copy.deepcopy(mlp)

        torch.save(best_mlp.state_dict(), model_path + '_' + str(idx) + '.mdl')

        null_res = dict(loss=list(), corr=list(), r2=list())
        for i in tqdm(range(n_shuff)):
            np.random.shuffle(y)
            tensorboard_path = os.path.join('tensorboard5/', str(idx) + '_' + str(int(round(time.time() % 1, 5) * 1e5)))
            _, shuf_r, _ = run_deep(X, y, idx, 0.2,
                                    train_batch_size=10, val_batch_size=5,
                                    learning_rate=1e-3,
                                    tensorboard_path=tensorboard_path,
                                    print_iteration=1,
                                    n_epochs=100,
                                    tensorboard=False,
                                    do_print=False)
            null_res['loss'].append(shuf_r['loss'])
            null_res['corr'].append(shuf_r['corr'])
            null_res['r2'].append(shuf_r['r2'])

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
