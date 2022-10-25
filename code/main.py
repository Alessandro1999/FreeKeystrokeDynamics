from typing import *
import torch
import torch.utils.data
import config
import net
import dataset
import data_preparation
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning as pl
import wandb

batch_size: int = 32
pl.seed_everything(config.seed)

wandb.init(project='FreeKeystrokeDynamics', entity='ale99')
logger = WandbLogger(name='long_run', project='FreeKeystrokeDynamics')

# wandb stuff
wandb.define_metric('epoch')
wandb.define_metric('train_loss', step_metric='epoch')
wandb.define_metric('val_loss', step_metric='epoch')
wandb.define_metric('train_acc', step_metric='epoch')
wandb.define_metric('val_acc', step_metric='epoch')

# Checkpoint to save the model with the lowest validation loss
checkpoint = callbacks.ModelCheckpoint("checkpoints/",
                                       monitor="val_loss",
                                       mode="min")

df = data_preparation.get_dataframes(
    config.ROOT_PATH.joinpath("data"))


datamodule = dataset.KeystrokeDataModule(df, test_perc=0.2,
                                         val_perc=0.125,
                                         batch_size=batch_size)
model = net.KeystrokeLSTM(50, 30,  50, 1)

trainer = pl.Trainer(max_epochs=30, gpus=1,
                     logger=logger, callbacks=[checkpoint])

trainer.test(model=model, datamodule=datamodule)
trainer.fit(model=model, datamodule=datamodule)

# load the model from checkpoint and test
model = net.KeystrokeLSTM.load_from_checkpoint(checkpoint.best_model_path)
model.first_test = False
trainer.test(model=model, datamodule=datamodule)

wandb.finish()
