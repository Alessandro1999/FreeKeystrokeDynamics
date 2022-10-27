from typing import *
import pandas as pd
import torch
import torch.utils.data as data
import pytorch_lightning as pl

import data_preparation
import config


class KeystrokeDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, keys: Set[str] = None):
        data = list(df.Timings)
        if keys is not None:  # we want to compute the vocabulary
            subjects: Set[str] = set(df.Subject)
            subjects.discard(config.UNK_SUB)
            config.subject_map: Dict[str, int] = {
                s: i+1 for i, s in enumerate(sorted(subjects))}
            config.subject_map[config.UNK_SUB] = 0

            config.key_map: Dict[str, int] = {
                k: i+2 for i, k in enumerate(sorted(keys))}
            config.key_map[config.PAD_KEY] = 0
            config.key_map[config.UNK_KEY] = 1

        ground_truth = torch.zeros(len(data))
        lengths = torch.zeros(len(data))
        keys_list = list()
        time_list = list()
        subjects = list(df.Subject)
        for i in range(len(data)):
            ground_truth[i] = config.subject_map[subjects[i]]
            lengths[i] = len(data[i])
            kt = torch.zeros(len(data[i]), dtype=torch.long)
            t = torch.zeros(len(data[i]), 2)
            for j, (k, press, release) in enumerate(data[i]):
                kt[j] = config.key_map.get(k, config.key_map[config.UNK_KEY])
                t[j][0] = press
                t[j][1] = release
            keys_list.append(kt)
            time_list.append(t)

        keys_tensor = torch.nn.utils.rnn.pad_sequence(
            keys_list, batch_first=True, padding_value=config.key_map[config.PAD_KEY])

        time_tensor = torch.nn.utils.rnn.pad_sequence(
            time_list, batch_first=True, padding_value=0.0)

        self.y = ground_truth
        self.k = keys_tensor
        self.t = time_tensor
        self.l = lengths

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        return self.y[index], self.k[index], self.t[index], self.l[index]


class KeystrokeDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, test_perc: float, val_perc: float, batch_size: int = 32):
        super().__init__()

        train_df, test_df = data_preparation.split_df_subjects(df,
                                                               1-test_perc)
        train_df, val_df = data_preparation.split_df_subjects(train_df,
                                                              1-val_perc)

        keys: Set[str] = data_preparation.get_keys(train_df)

        # define the datasets
        self.train_dataset: KeystrokeDataset = KeystrokeDataset(
            train_df, keys=keys)

        self.test_dataset: KeystrokeDataset = KeystrokeDataset(test_df)

        self.val_dataset: KeystrokeDataset = KeystrokeDataset(val_df)

        self.batch_size = batch_size

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
