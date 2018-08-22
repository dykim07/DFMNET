import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
from tqdm import tqdm

class DFMNET(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_lstm_layer=2, n_lstm_hidden=128, n_KDN_hidden=128):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_lstm_layer = n_lstm_layer
        self.n_lstm_hidden = n_lstm_hidden
        self.n_KDN_hidden = n_KDN_hidden
        self._build_model()

    def _build_model(self):

        self.SEN = nn.LSTM(
            input_size=self.n_input,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.n_lstm_layer,
            dropout=0.5
        )

        self.KDN = nn.Sequential(
            nn.Linear(self.n_lstm_hidden + self.n_input, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_output)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        r, _ = self.SEN(x.transpose(1, 0))
        r = r[-1, :, :]
        r = torch.cat([r, x[:, -1, :]], 1)
        y = self.KDN(r)

        return y
