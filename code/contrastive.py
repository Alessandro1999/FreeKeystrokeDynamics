import os
import re
import pandas as pd
from pathlib import Path
from typing import *
import csv
from tqdm import tqdm
import math
import random
import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks
import wandb

import config
import utils
from RandomLazyDataset import RandomLazySiameseDataset
from SiameseDataset import SiameseDataset
import dataloader
import net

seed = 17
pl.seed_everything(seed)

config.js_code_to_key, config.js_key_to_code = utils.get_keycode_mapping()

train_set : pd.DataFrame = utils.get_training_df(n = 68000)

perc: float = 0.000397

train_set, val_set = utils.train_val_split(df = train_set, perc = perc)

utils.compute_vocab(df = train_set)

train_batches_per_epoch : int = 150
batch_size : int = 512

# the list of the users for each dataset
train_users = sorted(set(train_set.PARTICIPANT_ID))
val_users = sorted(set(val_set.PARTICIPANT_ID))

# the training set will be shuffled
#train_siamese = ShuffleDataset(LazySiameseDataset(train_users,char_vocab), buffer_size = batch_size)
train_siamese = RandomLazySiameseDataset(train_users,config.char_vocab, train_set, max_len = train_batches_per_epoch * batch_size)
# for the validation it is not necessary
val_siamese = SiameseDataset(val_set,config.char_vocab)

datamodule = dataloader.KeystrokeDataModule(train_siamese,val_siamese)
model = net.KeystrokeLSTM(embedding_dim = 50, time_dim = 50, hidden_size = 25, output_size = 10, alpha = 1)

wandb.init(project='FreeKeystrokeDynamics', entity='ale99')
logger = pl.loggers.WandbLogger(name='long_run', project='FreeKeystrokeDynamics')

wandb.define_metric('epoch')
wandb.define_metric('train_loss',step_metric='epoch')
wandb.define_metric('val_loss',step_metric='epoch')

checkpoint = callbacks.ModelCheckpoint(config.ROOT_PATH.joinpath("checkpoints/"),
                                       monitor="val_loss",
                                       mode="min")

trainer = pl.Trainer(max_epochs=2,
                         accelerator='gpu',
                         devices=1,
                         logger=logger,
                         callbacks=[checkpoint])

trainer.fit(model=model, datamodule=datamodule)