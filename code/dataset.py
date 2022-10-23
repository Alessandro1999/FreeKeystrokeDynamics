from typing import *
import pandas as pd
import torch
import torch.utils.data as data
import pytorch_lightning as pl


class KeystrokeDataset(data.Dataset):
    def __init__(self,
                 ground_truth: torch.tensor,
                 keys: torch.tensor,
                 timings: torch.tensor,
                 lenghts: torch.tensor) -> None:
        super().__init__()
        self.y = ground_truth
        self.k = keys
        self.t = timings
        self.l = lenghts

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        return self.y[index], self.k[index], self.t[index], self.l[index]


class KeystrokeDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        # TODO see how to split the dataframe without having samples in the same date in 2 different dataset
