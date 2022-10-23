from typing import *
import torch
import torch.utils.data
import config
import net
import data_preparation
import pandas as pd
from tqdm import tqdm

df, keys = data_preparation.get_dataframes(
    config.ROOT_PATH.joinpath("data"))


dataset = data_preparation.pd_to_dataset(df, keys)

model = net.KeystrokeLSTM(10, 10, 1)

dl = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=32)

opt = torch.optim.SGD(model.parameters(), lr=0.01)


def train(epochs: int):
    avg_loss, batches = 0, 1
    epoch = 0
    for epoch in range(epochs):
        avg_loss: float = 0
        batches: int = 0
        for (ground_truth, keys, timings, lenghts) in dl:
            o: Dict[str, torch.tensor] = model(
                keys, timings, lenghts, ground_truth)

            loss: torch.tensor = o["loss"]
            avg_loss += loss.item()
            batches += 1

            loss.backward()

            opt.step()

            opt.zero_grad()

        print(f"The loss of epoch {epoch} is {round(avg_loss/batches,3)}")
        #avg_loss, batches = 0, 0
