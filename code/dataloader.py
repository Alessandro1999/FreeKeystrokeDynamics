from typing import *
import torch
import pytorch_lightning as pl

class KeystrokeDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_set : torch.utils.data.Dataset,
                 val_set : torch.utils.data.Dataset,
                 batch_size : int = 512):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_set,
                                           batch_size = self.batch_size,
                                           collate_fn = collate_fn, 
                                           shuffle = True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_set,
                                           batch_size = self.batch_size)



def collate_fn(batch : List[Dict[str,torch.Tensor]]) -> Dict[str,torch.Tensor]:
    batch_out : Dict[str,torch.Tensor] = dict()
    timings1 : List[torch.Tensor] = list()
    timings2 : List[torch.Tensor] = list()
    genuine : torch.Tensor = torch.zeros(len(batch))
    lengths1 : torch.Tensor = torch.zeros_like(genuine)
    lengths2 : torch.Tensor = torch.zeros_like(genuine)
    for i,sample in enumerate(batch):
        timings1.append(sample["timings1"])
        timings2.append(sample["timings2"])
        lengths1[i] = sample['length1']
        lengths2[i] = sample['length2']
        genuine[i] = sample['genuine']
    
    batch_out["genuine"] = genuine
    batch_out["lengths1"] = lengths1
    batch_out["lengths2"] = lengths2
    batch_out["timings1"] = torch.nn.utils.rnn.pad_sequence(timings1, batch_first = True)
    batch_out["timings2"] = torch.nn.utils.rnn.pad_sequence(timings2, batch_first = True)

    return batch_out
