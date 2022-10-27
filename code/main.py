from typing import *
import torch
import torch.utils.data
import config
import net
import dataset
import data_preparation
import take_sample
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning as pl
import wandb

train_params: Dict[str, Any] = {"epochs": 2,
                                "batch_size": 32,
                                "test_perc": 0.2,  # 20% of the original data
                                "val_perc": 0.125,  # 10% of the original data
                                "embedding_dim": 100,
                                "time_dim": 100,
                                "hidden_dim": 100,
                                "lstm_layers": 1
                                }


def train(epochs: int,
          batch_size: int,
          test_perc: float,
          val_perc: float,
          embedding_dim: int,
          time_dim: int,
          hidden_dim: int,
          lstm_layers: int) -> pl.LightningModule:
    pl.seed_everything(config.seed)

    wandb.init(project='FreeKeystrokeDynamics', entity='ale99')
    logger = WandbLogger(name='long_run', project='FreeKeystrokeDynamics')

    # wandb stuff
    wandb.define_metric('epoch')
    wandb.define_metric('threshold_a')
    wandb.define_metric('threshold_b')
    wandb.define_metric('train_loss', step_metric='epoch')
    wandb.define_metric('val_loss', step_metric='epoch')
    wandb.define_metric('test_a_loss', step_metric='epoch')
    wandb.define_metric('test_b_loss', step_metric='epoch')
    wandb.define_metric('FAR_b', step_metric='threshold_b')
    wandb.define_metric('FRR_b', step_metric='threshold_b')
    wandb.define_metric('FAR_a', step_metric='threshold_a')
    wandb.define_metric('FRR_a', step_metric='threshold_a')

    # Checkpoint to save the model with the lowest validation loss
    checkpoint = callbacks.ModelCheckpoint("checkpoints/",
                                           monitor="val_loss",
                                           mode="min")

    df = data_preparation.get_dataframes(
        config.ROOT_PATH.joinpath("data"))

    datamodule = dataset.KeystrokeDataModule(df,
                                             test_perc=test_perc,
                                             val_perc=val_perc,
                                             batch_size=batch_size)
    model = net.KeystrokeLSTM(embedding_dim,
                              time_dim,
                              hidden_dim,
                              lstm_layers)

    trainer = pl.Trainer(max_epochs=epochs,
                         gpus=1,
                         logger=logger,
                         callbacks=[checkpoint])

    trainer.test(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)

    # load the model from checkpoint and test
    model = net.KeystrokeLSTM.load_from_checkpoint(checkpoint.best_model_path)
    model.first_test = False
    trainer.test(model=model, datamodule=datamodule)

    wandb.finish()

    return model, config.key_map, config.subject_map


def predict(model: net.KeystrokeLSTM, key_map: Dict[str, int], subject_map: Dict[str, int]):
    df: pd.DataFrame = take_sample.take_sample(
        n=1, free=False, save_to_file=False)
    df = data_preparation.pd_conversion(df)  # compute metrics
    config.key_map = key_map
    d = dataset.KeystrokeDataset(df)
    claim, keys, timings, lenght = d[0]
    claim = claim.unsqueeze(0)
    keys = keys.unsqueeze(0)
    timings = timings.unsqueeze(0)
    lenght = lenght.unsqueeze(0)
    model.eval()
    out: torch.Tensor = model(keys,
                              timings,
                              lenght)["probabilities"].squeeze(0)

    reverse_subject_map = {i: s for s, i in subject_map.items()}
    out_prob: Dict[str, float] = dict()
    for i in range(len(out)):
        out_prob[reverse_subject_map.get(i, config.UNK_SUB)] = out[i].item()
    print(out_prob)
    print(f"I think you are: {reverse_subject_map[out.argmax().item()]}")
