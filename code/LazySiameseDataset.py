from typing import *
import torch
import pandas as pd
import random

import utils
import config

class LazySiameseDataset(torch.utils.data.IterableDataset):
    def __init__(self, users_list : List[int], vocab : Dict[str,int]):
        self.vocab = vocab
        self.users_list = users_list
        users_num: int = len(users_list) # each user file has 15 samples
        self.samples_num: int = 15 * len(users_list)
        # number of possible couples
        self.comb_num: int = self.samples_num * (self.samples_num-1) // 2
    
    def __iter__(self) -> Tuple[int,int]:
        return map(self.load_couple,map(self.internal_index_2_combination, range(self.comb_num)))

    def internal_index_2_combination(self,idx : int) -> Tuple[int,int]:
        return utils.index_2_combination(idx, self.samples_num, self.comb_num)

    def load_item(self, idx : int) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        file_idx : int = idx // 15
        row_idx : int= idx % 15
        user_id : int = self.users_list[file_idx]
        df : pd.DataFrame = pd.read_csv(f"data/Keystrokes_processed/{user_id}_keystrokes.txt",
                                sep=",",
                                names = config.column_names,
                                header=None,
                                encoding = "ISO-8859-1",
                                )
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

class ShuffleDataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset : torch.utils.data.IterableDataset , buffer_size : int):
    super().__init__()
    self.dataset = dataset
    self.buffer_size = buffer_size

  def __iter__(self):
    shufbuf = []
    try:
      dataset_iter = iter(self.dataset)
      for i in range(self.buffer_size): # iterate over the dataset for buffer times
        shufbuf.append(next(dataset_iter)) # append the data to the shuffle buffer
    except:
      self.buffer_size = len(shufbuf)

    try:
      while True:
        try:
          item = next(dataset_iter) # take a new sample
          evict_idx = random.randint(0, self.buffer_size - 1) # take a random index
          yield shufbuf[evict_idx] # return a random element of the shuffle buffer
          shufbuf[evict_idx] = item # substitute the element with new one
        except StopIteration: # when the iterator ends
          break
      while len(shufbuf) > 0: # return the remaining element of the shuffle buffer
        yield shufbuf.pop()
    except GeneratorExit:
      pass