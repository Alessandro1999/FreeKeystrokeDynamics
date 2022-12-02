from typing import *
import torch
import pandas as pd
from tqdm import tqdm

import utils

class SiameseDataset(torch.utils.data.Dataset):
    '''
    Dataset for the Siamese training of Keystroke Dynamics
    '''
    def __init__(self, df : pd.DataFrame, vocab : Dict[str,int]):
        # the identities of the users
        self.ground_truth : torch.Tensor = torch.tensor(list(df.PARTICIPANT_ID))
        self.tot = self.ground_truth.shape[0] # number of samples
        # we are doing siamese training, so the total n will be the possible couples
        self.n = int((self.tot-1)*self.tot / 2) # number of combinations
        # lengths and timings
        self.lengths : torch.Tensor = torch.zeros_like(self.ground_truth)
        strings = list(df.TIMINGS)
        tensors : List[torch.Tensor] = list()
        for i,string in tqdm(enumerate(strings),total = self.tot): # for every row of the dataset
            tensors.append(utils.string_to_tensor(string,vocab)) # convert the timings into tensors
            self.lengths[i] = tensors[-1].shape[0] # get the lenght of the sequence
        # pad the smaller sequences with 0's
        self.timings : torch.Tensor = torch.nn.utils.rnn.pad_sequence(tensors,batch_first=True)
    
    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self,idx) ->Tuple[torch.Tensor]:
        # convert the i-th element in the 2 indexes of the couple
        idx1, idx2 = utils.index_2_combination(idx,self.tot,self.n)
        # return the data
        return {
                "genuine" : (self.ground_truth[idx1] == self.ground_truth[idx2]).long(),
                "timings1": self.timings[idx1],
                "lengths1": self.lengths[idx1],
                "timings2": self.timings[idx2],
                "lengths2": self.lengths[idx2]
                }
    
