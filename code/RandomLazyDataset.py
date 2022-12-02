from typing import *
import torch
import pandas as pd
import random

import utils

class RandomLazySiameseDataset(torch.utils.data.Dataset):
    def __init__(self, users_list : List[int], vocab : Dict[str,int], dataset : pd.DataFrame, max_len : int = None):
        self.vocab = vocab
        self.users_list = users_list
        self.dataset = dataset
        users_num: int = len(users_list) # each user file has 15 samples
        self.samples_num: int = 15 * len(users_list)
        self.len = max_len if max_len is not None else self.samples_num
        # number of possible couples
        #self.comb_num: int = self.samples_num * (self.samples_num-1) // 2

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx : int) -> Dict[str,torch.Tensor]:
        # whether to pick a genuine or a false twin
        pick_genuine = bool(random.randint(0,1)) 

        # we want a genuine couple
        if pick_genuine:
            sample_idx = idx % 15
            twin_idx = sample_idx
            # sample until you pick a different twin than itself
            while twin_idx == sample_idx:
                twin_idx = random.randint(0,14)
            final_idx = (idx - sample_idx) + twin_idx
        # we want a non-genuine couple
        else:
            # since we have only 14 out of 1020000 possibility of picking a genuine
            # we do not really care of not considering them in the sampling
            final_idx = idx
            while final_idx == idx:
                final_idx = random.randint(0,self.samples_num - 1)
        return self.load_couple((idx,final_idx))

    def load_item(self, idx : int) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        file_idx : int = idx // 15
        row_idx : int= idx % 15
        user_id : int = self.users_list[file_idx]
        #df : pd.DataFrame = pd.read_csv(f"data/Keystrokes_processed/{user_id}_keystrokes.txt",
        #                        sep=",",
        #                        names = column_names,
        #                        header=None,
        #                        encoding = "ISO-8859-1",
        #                        )
        df = self.dataset[self.dataset.PARTICIPANT_ID == user_id]
        row = df.iloc[row_idx]
        ground_truth : torch.Tensor = torch.tensor(row.PARTICIPANT_ID)
        timings : torch.Tensor = utils.string_to_tensor(row.TIMINGS, self.vocab)
        length : torch.Tensor = torch.tensor(timings.shape[0])
        return ground_truth, timings, length
    
    def load_couple(self, indexes: Tuple[int,int]) -> Dict[str,torch.Tensor]:
        g1, t1, l1 = self.load_item(indexes[0])
        g2, t2, l2 = self.load_item(indexes[1])
        return {
            "genuine" : g1 == g2,
            "timings1": t1,
            "length1" : l1,
            "timings2": t2,
            "length2" : l2
            }