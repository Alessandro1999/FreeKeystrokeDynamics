from cProfile import label
from typing import *
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb

import config


class KeystrokeLSTM(pl.LightningModule):
    def __init__(self, embedding_dim: int, time_dim: int, hidden_size: int, lstm_layers: int) -> None:
        super().__init__()

        # embedding layer
        self.key_emb = torch.nn.Embedding(num_embeddings=len(config.key_map),
                                          embedding_dim=embedding_dim,
                                          padding_idx=config.key_map[config.PAD_KEY])

        # linear projection of the time features
        self.time_features = torch.nn.Linear(2, time_dim)

        # lstm
        self.lstm = torch.nn.LSTM(input_size=embedding_dim+time_dim,
                                  hidden_size=hidden_size,
                                  num_layers=lstm_layers)
        # linear layer
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=len(config.subject_map))

        # activation and loss
        self.softmax = torch.nn.Softmax(dim=-1)
        self.loss = torch.nn.CrossEntropyLoss()

        # boolean flag that just tells us if we have never tested the model yet
        self.first_test: bool = True

        # the thresholds for which we will compute FAR and FRR
        self.thresholds = torch.arange(0, 1, 0.001, device=self.device)
        self.false_acceptances = torch.zeros_like(
            self.thresholds, device=self.device)
        self.false_rejections = torch.zeros_like(
            self.thresholds, device=self.device)
        self.false_claims: int = 0
        self.genuine_claims: int = 0

        self.save_hyperparameters()

    def forward(self, keys: torch.Tensor, timings: torch.Tensor, lenghts: torch.Tensor, y: torch.Tensor = None):
        batch_size = lenghts.shape[0]

        emb = self.key_emb(keys)
        timings = self.time_features(timings)

        x = torch.concat((emb, timings), dim=-1)

        x = self.lstm(x)[0][torch.arange(batch_size), (lenghts-1).long(), :]

        logits = self.linear(x)

        probabilities = self.softmax(logits)

        out = {"logits": logits, "probabilities": probabilities}
        if y is not None:
            out["loss"] = self.loss(logits, y.long())

        return out

    def step(self, batch) -> torch.Tensor:
        y, keys, timings, lenghts = batch
        out: Dict[str, torch.tensor] = self(keys, timings, lenghts, y)
        loss = out["loss"]
        return loss

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        return self.step(train_batch)

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        return self.step(val_batch)

    def test_step(self, test_batch, batch_idx) -> torch.Tensor:
        y, keys, timings, lenghts = test_batch
        batch_size: int = y.shape[0]
        # output probabilities of the model
        out: Dict[str, torch.tensor] = self(keys, timings, lenghts, y)
        # the boolean mask for the probes belonging to unknown users
        unk_y: torch.Tensor = (y == config.subject_map[config.UNK_SUB])
        # the number of non enrolled users in this batch
        non_enrolled_users: int = unk_y.sum().item()
        # the number of enrolled users in this batch
        enrolled_users: int = (batch_size - non_enrolled_users)
        # the number of all enrolled users
        all_users_num: int = len(config.known_subject)
        # every enrolled user can be a genuine claim
        self.genuine_claims += enrolled_users
        # every enrolled user can have n-1 false claims, every non enrolled can have n insted
        # n is the number of enrolled subjects
        self.false_claims += (enrolled_users * (all_users_num-1)) +\
                             (non_enrolled_users * all_users_num)

        if self.thresholds.device != self.device:
            self.thresholds = self.thresholds.to(self.device)
        if self.false_acceptances.device != self.device:
            self.false_acceptances = self.false_acceptances.to(self.device)
        if self.false_rejections != self.device:
            self.false_rejections = self.false_rejections.to(self.device)

        genuin_claims = (out["probabilities"]
                         # take only probability for the real identity
                         [torch.arange(batch_size), y.long()]
                         # take only enrolled subjects
                         [torch.logical_not(unk_y)]).to(self.device)

        false_claims = out["probabilities"] + 0  # make a copy
        # do not consider genuine claims by setting them to -1
        false_claims[torch.arange(batch_size), y.long()] = -1
        # you cannot claim the identity of the unknown subject
        false_claims[:, config.subject_map[config.UNK_SUB]] = -1

        # for every threshold
        for t in range(len(self.thresholds)):
            # this iteration's threshold
            threshold: float = self.thresholds[t].item()

            self.false_rejections[t] += (genuin_claims < threshold).sum()

            self.false_acceptances[t] += (false_claims >= threshold).sum()

        return out["loss"]

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

    def test_epoch_end(self, outputs) -> None:
        loss = sum(outputs) / len(outputs)

        log_detail: str = 'before training' if self.first_test else 'after training'
        fars: torch.Tensor = self.false_acceptances / self.false_claims
        frrs: torch.Tensor = self.false_rejections / self.genuine_claims

        argmin_eer: int = int(torch.argmin(torch.abs(fars - frrs)).item())
        approx_eer: float = ((fars[argmin_eer] + frrs[argmin_eer])/2).item()
        eer_t: float = self.thresholds[argmin_eer].item()

        self.false_acceptances = torch.zeros_like(self.false_acceptances)
        self.false_rejections = torch.zeros_like(self.false_acceptances)
        self.false_claims = 0
        self.genuine_claims = 0

        plt.plot(self.thresholds.cpu().numpy(),
                 fars.cpu().numpy(), label="FRR")
        plt.plot(self.thresholds.cpu().numpy(),
                 frrs.cpu().numpy(), label="FAR")
        plt.plot(eer_t, approx_eer, 'ko', label="EER")
        plt.xlabel("Threshold")
        plt.ylabel("FAR/FRR")
        plt.legend()
        wandb.log({f"EER threshold {log_detail}": eer_t})
        wandb.log({f"EER {log_detail}": approx_eer})
        wandb.log(
            {f"FAR/FRR {log_detail}": plt})

        self.log_metrics(loss.item(), f'test {log_detail}')

        if self.first_test:
            self.first_test = False
        return super().test_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
