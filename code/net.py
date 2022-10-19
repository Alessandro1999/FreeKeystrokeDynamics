from turtle import forward
import config
import torch


class KeystrokeLSTM(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, lstm_layers: int) -> None:
        super().__init__()

        # embedding layer
        self.key_emb = torch.nn.Embedding(num_embeddings=len(config.key_map),
                                          embedding_dim=embedding_dim,
                                          padding_idx=config.key_map[config.PAD_KEY])
        # lstm
        self.lstm = torch.nn.LSTM(input_size=embedding_dim+2,
                                  hidden_size=hidden_size,
                                  num_layers=lstm_layers)
        # linear layer
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=len(config.subject_map))

        # activation and loss
        self.softmax = torch.nn.Softmax(dim=-1)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, keys: torch.tensor, timings: torch.tensor, lenghts: torch.tensor, y: torch.tensor = None):
        emb = self.key_emb(keys)

        x = torch.concat((emb, timings), dim=-1)

        x = self.lstm(x)[0][:, -1, :]

        logits = self.linear(x)

        probabilities = self.softmax(logits)

        out = {"logits": logits, "probabilities": probabilities}
        if y is not None:
            out["loss"] = self.loss(logits, y.long())

        return out
