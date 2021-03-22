import torch
import numpy as np
import json
import os
import argparse
import random
import pickle
from collections import defaultdict
import itertools

from data_new import DataProvider
import train_ceae
from copy import deepcopy


def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def load_pickle(pickle_file):
    data = []
    with open(pickle_file, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    return data


def wrap_training_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])

    return aux_dict


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')


def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])


def main(train_num_epochs=10000):
    train_fn = train_ceae.train_ceae
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training_params = {
        'device': device,
        'model_save_folder': os.path.join('model_save', 'ceae'),
        'train_num_epochs': train_num_epochs,
        'batch_size': 64,
        'lr': 1e-4
    }

    safe_make_dir(training_params['model_save_folder'])

    data_provider = DataProvider(batch_size=training_params['batch_size'])
    training_params.update(
        {
            'input_dim': data_provider.shape_dict['prot'],
            'output_dim': data_provider.shape_dict['target']
        }
    )

    set_random_seed(2021)
    # start unlabeled training
    _, historys = train_fn(
        dataloder=data_provider.get_unlabeld_dataloader(),
        kwargs = training_params
    )

    with open(os.path.join(training_params['model_save_folder'], f'pre_train_history.pickle'), 'wb') as f:
        for history in historys:
            pickle.dump(dict(history), f)


if __name__ == '__main__':
    train_num_epochs = 10
    main(train_num_epochs=train_num_epochs)
