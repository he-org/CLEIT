import os
import datetime
import numpy as np
import random
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold

_RNG_SEED = None


def filter_features(df, mean_tres=1.0, std_thres=0.5, cor_thres=None):
    """
    filter genes of low information burden
    first need to reverse transformation applied log2(x+0.001)
    mean: sparsity threshold, std: variability threshold
    :param df: samples X features
    :param mean_tres:
    :param std_thres:
    :param cor_thres:
    :return:
    """
    df = df.loc[:, df.apply(lambda col: col.isna().sum()) == 0]
    feature_stds = df.std()
    feature_means = df.mean()
    std_to_drop = feature_stds[list(np.where(feature_stds <= std_thres)[0])].index.tolist()
    mean_to_drop = feature_means[list(np.where(feature_means <= mean_tres)[0])].index.tolist()
    to_drop = list(set(std_to_drop) | set(mean_to_drop))
    df.drop(labels=to_drop, axis=1, inplace=True)
    return df


def filter_with_MAD(df, k=5000):
    result = df[(df - df.median()).abs().median().nlargest(k).index.tolist()]
    return result


def filter_with_uqstd(df, k=1000):
    uq_perc = df.apply(lambda col: len(col.unique())) / df.shape[0]
    stds = df.std()
    features = (uq_perc * stds).nlargest(k).index.tolist()
    result = df[features]
    return result


def align_feature(ref_df, dest_df):
    matched_features = list(set(ref_df.columns.tolist()) & set(dest_df.columns.tolist()))
    matched_features.sort()
    print('Aligned dataframes have {} features in common'.format(len(matched_features)))
    return dest_df[matched_features]


def get_rng(obj=None):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
    Get a good RNG seeded with time, pid and the object.
    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return random.Random(seed)


class DataProvider:
    def __init__(self, filter=None, batch_size=64, target='AUC', random_seed=2021):
        self.seed = random_seed
        self.target = target
        self.batch_size = batch_size
        self.filter = filter
        self._load_trans_data()
        self._load_prot_data()
        self._load_target_data()
        self.shape_dict = {'trans': self.trans_dat.shape[-1],
                           'prot': self.prot_dat.shape[-1],
                           'target': self.target_df.shape[-1]}

    def _load_trans_data(self):
        self.trans_dat = pd.read_csv('./data/ccle_pro_trans_pic50/adjusted_ccle_tcga_ad_tpm_log2.csv', index_col=0)
        self.trans_repr_dat = pd.read_csv('./data/ccle_pro_trans_pic50/hidden_repr_cells_v7_0325.csv', index_col=0)

    def _load_prot_data(self):
        self.prot_dat = pd.read_csv('./data/ccle_pro_trans_pic50/tri_basal_proteomics.csv', index_col=0)
        self.prot_dat.dropna(axis=1, inplace=True)
        # self.prot_dat = self.ccle_prot_dat.append(self.tcga_prot_dat)
        # self.prot_dat.dropna(axis=1, inplace=True)
        # self.prot_dat = self.prot_dat.reset_index().groupby('index').mean()
        # if self.filter is not None:
        #     if self.filter == 'align':
        #         self.prot_dat = align_feature(ref_df=self.trans_dat, dest_df=self.prot_dat)
        #     elif self.filter == 'mad':
        #         self.ccle_prot_dat = filter_with_MAD(df=self.ccle_prot_dat, k=1000)
        #         # self.tcga_prot_dat = filter_with_MAD(df=self.tcga_prot_dat, k=1000)
        #         self.prot_dat = self.ccle_prot_dat.append(self.tcga_prot_dat)
        #         self.prot_dat.dropna(axis=1, inplace=True)
        #         self.prot_dat = self.prot_dat.reset_index().groupby('index').mean()
        #     elif self.filter == 'uq':
        #         self.ccle_prot_dat = filter_with_uqstd(df=self.ccle_prot_dat, k=1000)
        #         # self.tcga_prot_dat = filter_with_MAD(df=self.tcga_prot_dat, k=1000)
        #         self.prot_dat = self.ccle_prot_dat.append(self.tcga_prot_dat)
        #         self.prot_dat.dropna(axis=1, inplace=True)
        #         self.prot_dat = self.prot_dat.reset_index().groupby('index').mean()
        #     else:
        #         pass

    def _load_target_data(self):

        target_df = pd.read_csv('./data/ccle_pro_trans_pic50/ccle_gdsc_ctrp_drug_response_pic50_370_369.csv',
                                index_col=0)
        target_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        target_df = target_df.transpose()
        target_df.drop(columns=target_df.columns[
            target_df.isna().sum() / len(target_df) >= 0.1], inplace=True)
        self.target_df = target_df

    #
    # def get_unlabeled_trans_dataloader(self):
    #     trans_dataset = TensorDataset(torch.from_numpy(self.trans_dat.values.astype('float32')))
    #     unlabeled_trans_dataloader = DataLoader(trans_dataset,
    #                                           batch_size=self.batch_size,
    #                                           shuffle=True)
    #
    #     return unlabeled_trans_dataloader

    def get_unlabeld_dataloader(self):
        matched_samples = self.trans_repr_dat.index.intersection(self.prot_dat.index)
        matched_dataset = TensorDataset(
            torch.from_numpy(self.prot_dat.loc[matched_samples].values.astype('float32')),
            torch.from_numpy(self.trans_repr_dat.loc[matched_samples].values.astype('float32'))
        )
        unlabeled_dataloader = DataLoader(matched_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True
                                          )
        return unlabeled_dataloader

    def get_labeled_data_generator(self, omics='prot'):
        labeled_samples = self.trans_dat.index.intersection(self.target_df.index)
        labeled_samples = self.prot_dat.index.intersection(labeled_samples)
        labeled_target_df = self.target_df.loc[labeled_samples]
        labeled_samples = labeled_samples[labeled_target_df.shape[1] - labeled_target_df.isna().sum(axis=1) >= 2]
        labeled_target_df = self.target_df.loc[labeled_samples]

        sample_label_vec = (
                labeled_target_df.isna().sum(axis=1) <= labeled_target_df.isna().sum(axis=1).median()).astype('int')
        s_kfold = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)

        if omics == 'trans':
            for train_index, test_index in s_kfold.split(self.trans_dat.loc[labeled_samples].values,
                                                         sample_label_vec):
                train_labeled_df, test_labeled_df = self.trans_dat.loc[labeled_samples].values[train_index], \
                                                    self.trans_dat.loc[labeled_samples].values[test_index]
                train_labels, test_labels = labeled_target_df.values[train_index].astype('float32'), \
                                            labeled_target_df.values[
                                                test_index].astype('float32')

                train_labeled_dateset = TensorDataset(
                    torch.from_numpy(train_labeled_df.astype('float32')),
                    torch.from_numpy(train_labels))
                test_labeled_dateset = TensorDataset(
                    torch.from_numpy(test_labeled_df.astype('float32')),
                    torch.from_numpy(test_labels))

                train_labeled_dataloader = DataLoader(train_labeled_dateset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True, drop_last=True)

                test_labeled_dataloader = DataLoader(test_labeled_dateset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True)

                yield train_labeled_dataloader, test_labeled_dataloader
        else:
            for train_index, test_index in s_kfold.split(self.prot_dat.loc[labeled_samples].values,
                                                         sample_label_vec):
                train_labeled_df, test_labeled_df = self.prot_dat.loc[labeled_samples].values[train_index], \
                                                    self.prot_dat.loc[labeled_samples].values[test_index]
                train_labels, test_labels = labeled_target_df.values[train_index].astype('float32'), \
                                            labeled_target_df.values[
                                                test_index].astype('float32')

                train_labeled_dateset = TensorDataset(
                    torch.from_numpy(train_labeled_df.astype('float32')),
                    torch.from_numpy(train_labels))
                test_labeled_dateset = TensorDataset(
                    torch.from_numpy(test_labeled_df.astype('float32')),
                    torch.from_numpy(test_labels))

                train_labeled_dataloader = DataLoader(train_labeled_dateset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True, drop_last=True)

                test_labeled_dataloader = DataLoader(test_labeled_dateset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True)

                yield train_labeled_dataloader, test_labeled_dataloader

    def get_labeled_trans_dataloader(self):
        trans_labeled_samples = self.trans_dat.index.intersection(self.target_df.index)
        trans_target_df = self.target_df.loc[trans_labeled_samples]
        trans_labeled_samples = trans_labeled_samples[
            trans_target_df.shape[1] - trans_target_df.isna().sum(axis=1) >= 2]
        trans_target_df = self.target_df.loc[trans_labeled_samples]

        labeled_trans_dataset = TensorDataset(
            torch.from_numpy(self.trans_dat.loc[trans_labeled_samples].values.astype('float32')),
            torch.from_numpy(trans_target_df.values.astype('float32'))
        )
        labeled_trans_dataloader = DataLoader(labeled_trans_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True)
        return labeled_trans_dataloader

    def get_labeled_prot_dataloader(self):
        trans_labeled_samples = self.trans_dat.index.intersection(self.target_df.index)
        prot_labeled_samples = self.trans_dat.index.intersection(self.target_df.index)
        prot_only_labeled_samples = prot_labeled_samples.difference(trans_labeled_samples)
        prot_labeled_samples = prot_labeled_samples.difference(prot_only_labeled_samples)

        prot_target_df = self.target_df.loc[prot_labeled_samples]
        prot_labeled_samples = prot_labeled_samples[prot_target_df.shape[1] - prot_target_df.isna().sum(axis=1) >= 2]
        prot_target_df = self.target_df.loc[prot_labeled_samples]

        labeled_prot_dataset = TensorDataset(
            torch.from_numpy(self.prot_dat.loc[prot_labeled_samples].values.astype('float32')),
            torch.from_numpy(prot_target_df.values.astype('float32'))
        )
        labeled_prot_dataloader = DataLoader(labeled_prot_dataset,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             drop_last=True)

        return labeled_prot_dataloader

    def get_labeled_trans_dataloader_generator(self, drug=None):
        # drug = DRUG_DICT[drug]
        # drug_target_df = self.target_df[drug]
        # drug_target_df.dropna(inplace=True)
        # drug_trans_labeled_samples = self.trans_dat.index.intersection(drug_target_df.index)
        # # get trans dataset and dataloader
        # drug_trans_target_df = drug_target_df.loc[drug_trans_labeled_samples]
        # trans_label_vec = (drug_trans_target_df < np.median(drug_trans_target_df)).astype('int')
        trans_labeled_samples = self.trans_dat.index.intersection(self.target_df.index)
        trans_target_df = self.target_df.loc[trans_labeled_samples]
        trans_labeled_samples = trans_labeled_samples[
            trans_target_df.shape[1] - trans_target_df.isna().sum(axis=1) >= 2]
        trans_target_df = self.target_df.loc[trans_labeled_samples]

        sample_label_vec = (trans_target_df.isna().sum(axis=1) <= trans_target_df.isna().sum(axis=1).median()).astype(
            'int')

        s_kfold = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)
        for train_index, test_index in s_kfold.split(self.trans_dat.loc[trans_labeled_samples].values,
                                                     sample_label_vec):
            train_labeled_df, test_labeled_df = self.trans_dat.loc[trans_labeled_samples].values[train_index], \
                                                self.trans_dat.loc[trans_labeled_samples].values[test_index]
            train_labels, test_labels = trans_target_df.values[train_index].astype('float32'), trans_target_df.values[
                test_index].astype('float32')

            train_labeled_dateset = TensorDataset(
                torch.from_numpy(train_labeled_df.astype('float32')),
                torch.from_numpy(train_labels))
            test_labeled_dateset = TensorDataset(
                torch.from_numpy(test_labeled_df.astype('float32')),
                torch.from_numpy(test_labels))

            train_labeled_dataloader = DataLoader(train_labeled_dateset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True, drop_last=True)

            test_labeled_dataloader = DataLoader(test_labeled_dateset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True)

            yield train_labeled_dataloader, test_labeled_dataloader

    def get_labeled_prot_dataloader_generator(self, drug=None):
        # drug = DRUG_DICT[drug]
        # drug_target_df = self.target_df[drug]
        # drug_target_df.dropna(inplace=True)
        # drug_trans_labeled_samples = self.trans_dat.index.intersection(drug_target_df.index)
        # drug_prot_labeled_samples = self.prot_dat.index.intersection(drug_target_df.index)
        # drug_prot_only_labeled_samples = drug_prot_labeled_samples.difference(drug_trans_labeled_samples)
        # drug_prot_labeled_samples = drug_prot_labeled_samples.difference(drug_prot_only_labeled_samples)
        #
        # drug_prot_target_df = drug_target_df.loc[drug_prot_labeled_samples]
        # prot_label_vec = (drug_prot_target_df < np.median(drug_prot_target_df)).astype('int')
        trans_labeled_samples = self.trans_dat.index.intersection(self.target_df.index)
        prot_labeled_samples = self.prot_dat.index.intersection(self.target_df.index)
        prot_only_labeled_samples = prot_labeled_samples.difference(trans_labeled_samples)
        prot_labeled_samples = prot_labeled_samples.difference(prot_only_labeled_samples)

        prot_target_df = self.target_df.loc[prot_labeled_samples]
        prot_labeled_samples = prot_labeled_samples[prot_target_df.shape[1] - prot_target_df.isna().sum(axis=1) >= 2]
        prot_target_df = self.target_df.loc[prot_labeled_samples]

        sample_label_vec = (prot_target_df.isna().sum(axis=1) <= prot_target_df.isna().sum(axis=1).median()).astype(
            'int')

        s_kfold = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)
        for train_index, test_index in s_kfold.split(self.prot_dat.loc[prot_labeled_samples].values,
                                                     sample_label_vec):
            train_labeled_df, test_labeled_df = self.prot_dat.loc[prot_labeled_samples].values[train_index], \
                                                self.prot_dat.loc[prot_labeled_samples].values[test_index]
            train_labels, test_labels = prot_target_df.values[train_index].astype('float32'), prot_target_df.values[
                test_index].astype('float32')

            train_labeled_dateset = TensorDataset(
                torch.from_numpy(train_labeled_df.astype('float32')),
                torch.from_numpy(train_labels))
            test_labeled_dateset = TensorDataset(
                torch.from_numpy(test_labeled_df.astype('float32')),
                torch.from_numpy(test_labels))

            train_labeled_dataloader = DataLoader(train_labeled_dateset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True, drop_last=True)

            test_labeled_dataloader = DataLoader(test_labeled_dateset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True)

            yield train_labeled_dataloader, test_labeled_dataloader


    def get_labeled_samples(self):
        labeled_samples = self.trans_dat.index.intersection(self.target_df.index)
        labeled_samples = self.prot_dat.index.intersection(labeled_samples)
        labeled_target_df = self.target_df.loc[labeled_samples]
        labeled_samples = labeled_samples[labeled_target_df.shape[1] - labeled_target_df.isna().sum(axis=1) >= 2]

        return labeled_samples
