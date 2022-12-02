from typing import *
import torch
import pytorch_lightning as pl

import config

class KeystrokeLSTM(pl.LightningModule):
    def __init__(self,
                 embedding_dim: int,
                 time_dim: int,
                 hidden_size: int,
                 output_size : int,
                 alpha : float = 1.5,
                 lstm_layers: int = 1) -> None:
        super().__init__()

        # embedding layer
        self.key_emb = torch.nn.Embedding(num_embeddings=len(config.char_vocab),
                                          embedding_dim=embedding_dim,
                                          padding_idx=config.char_vocab[config.PAD_KEY])

        # linear projection of the time features
        self.time_features = torch.nn.Linear(2, time_dim)

        # lstm
        self.lstm = torch.nn.LSTM(input_size=embedding_dim+time_dim,
                                  hidden_size=hidden_size,
                                  num_layers=lstm_layers)
        # linear layer
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=output_size)
        # activation function
        self.activation = torch.nn.functional.relu

        # loss hyperparam
        self.alpha = alpha
   
        self.save_hyperparameters()

    def single_forward(self,
                       timings: torch.Tensor,
                       lenghts: torch.Tensor) -> torch.Tensor:
        
        batch_size = lenghts.shape[0]

        emb = self.key_emb(timings[:,:,0].long())
        timings = self.time_features(timings[:,:,1:])

        x = torch.concat((emb, timings), dim=-1)

        x = self.lstm(x)[0][torch.arange(batch_size), (lenghts-1).long(), :]

        x = self.linear(x)

        x = self.activation(x)

        return x

    def forward(self,
                timings1: torch.Tensor,
                lengths1: torch.Tensor,
                timings2: torch.Tensor,
                lengths2: torch.Tensor,
                genuine: torch.Tensor
                ) -> torch.Tensor:

        o1 = self.single_forward(timings1,lengths1)
        o2 = self.single_forward(timings2,lengths2)

        euclidean_distance = (((o1 - o2)**2).sum(dim = 1)) ** 1/2

        return contrastive_loss(euclidean_distance,genuine, alpha = self.alpha)

    def step(self, batch) -> torch.Tensor:
        loss : torch.Tensor = self(**batch)
        return loss

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        return self.step(train_batch)

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        return self.step(val_batch)
    
    def log_metrics(self, loss: float, type: str):
        self.log(f'{type}_loss', loss)
        self.log(f'epoch', float(self.current_epoch))

    def training_epoch_end(self, outputs) -> None:
        loss = sum([x["loss"] for x in outputs]) / len(outputs)
        self.log_metrics(loss.item(), 'train')
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs) -> None:
        loss = sum(outputs) / len(outputs)
        self.log_metrics(loss.item(), 'val')
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def contrastive_loss(distances : torch.Tensor, genuine : torch.Tensor, alpha : float = 1.5) -> torch.Tensor:
        zero = torch.tensor(0)
        genuine_loss = genuine * (distances**2) / 2
        impostor_loss = (1 - genuine) * ((torch.maximum(zero,alpha - distances))**2) / 2
        return (impostor_loss + genuine_loss).mean()