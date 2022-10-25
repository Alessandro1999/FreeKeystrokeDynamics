from typing import *
import torch
import pytorch_lightning as pl

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

        # logging stuff
        self.train_correct: int = 0
        self.val_correct: int = 0
        self.test_correct: int = 0

        self.train_total: int = 0
        self.val_total: int = 0
        self.test_total: int = 0

        self.first_test: bool = True

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

    def step(self, batch) -> Tuple[torch.Tensor, int, int]:
        y, keys, timings, lenghts = batch
        out: Dict[str, torch.tensor] = self(keys, timings, lenghts, y)
        loss = out["loss"]
        correct = ((out["probabilities"].argmax(dim=1) == y)).sum().item()
        total = (y != -1).sum().item()
        return loss, correct, total

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        loss, correct, total = self.step(train_batch)
        self.train_correct += correct
        self.train_total += total
        return loss

    def validation_step(self, train_batch, batch_idx) -> torch.Tensor:
        loss, correct, total = self.step(train_batch)
        self.val_correct += correct
        self.val_total += total
        return loss

    def test_step(self, train_batch, batch_idx) -> torch.Tensor:
        loss, correct, total = self.step(train_batch)
        self.test_correct += correct
        self.test_total += total
        return loss

    def log_metrics(self, correct, total, loss, type: str):
        accuracy: float = correct / total
        self.log(f'{type}_acc', accuracy)
        self.log(f'{type}_loss', loss)
        self.log(f'epoch', float(self.current_epoch))

    def training_epoch_end(self, outputs) -> None:
        loss = sum([x["loss"] for x in outputs]) / len(outputs)
        self.log_metrics(self.train_correct, self.train_total,
                         loss.item(), 'train')
        self.train_correct, self.train_total = 0, 0
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs) -> None:
        loss = sum(outputs) / len(outputs)
        self.log_metrics(self.val_correct, self.val_total, loss.item(), 'val')
        self.val_correct, self.val_total = 0, 0
        return super().validation_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        loss = sum(outputs) / len(outputs)
        self.log_metrics(self.test_correct, self.test_total,
                         loss.item(), 'test_b' if self.first_test else 'test_a')
        if self.first_test:
            self.first_test = False
        self.test_correct, self.test_total = 0, 0
        return super().test_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)

    def predict(self,):
        pass  # TODO
