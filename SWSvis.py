import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error

from Utils.trainUtils import *
from dataset.DataLoader import DataLoader, DiabetesDataset
from Models.DFMNET import DFMNET as MODEL


# train paramters
INPUT_DIM = 20
OUTPUT_DIM = 39
CUDA_ID = 2
N_EPOCHS = 30
BATCH_SIZE = 512
LR = 0.001

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
#
# class DFMNET(nn.Module):
#     def __init__(self, n_input: int, n_output: int, n_lstm_layer=2, n_lstm_hidden=128, n_KDN_hidden=128, lr=0.001,
#                  n_epochs=5, batch_size=16):
#         super().__init__()
#         self.n_input = n_input
#         self.n_output = n_output
#         self.n_lstm_layer = n_lstm_layer
#         self.n_lstm_hidden = n_lstm_hidden
#         self.n_KDN_hidden = n_KDN_hidden
#         self.lr = lr
#         self.n_epochs = n_epochs
#         self.batch_size = batch_size
#
#         self.n_hidden_1 = n_KDN_hidden
#         self.n_hidden_2 = n_KDN_hidden
#         self.n_hidden_3 = n_KDN_hidden
#         self.n_hidden_4 = n_KDN_hidden
#         self.n_hidden_5 = n_KDN_hidden
#
#         self._build_model()
#
#
#     def _build_model(self):
#         self.SEN = nn.LSTM(
#             input_size=self.n_input,
#             hidden_size=self.n_lstm_hidden,
#             num_layers=self.n_lstm_layer,
#             dropout=0.5
#         )
#
#         self.KDN = nn.Sequential(
#             nn.Linear(self.n_lstm_hidden + self.n_input, self.n_hidden_1),
#             nn.ReLU(),
#
#             nn.Linear(self.n_hidden_1, self.n_hidden_2),
#             nn.ReLU(),
#
#             nn.Linear(self.n_hidden_2, self.n_hidden_3),
#             nn.ReLU(),
#
#             nn.Linear(self.n_hidden_3, self.n_hidden_4),
#             nn.ReLU(),
#
#             nn.Linear(self.n_hidden_4, self.n_hidden_5),
#             nn.ReLU(),
#
#             nn.Linear(self.n_hidden_5, self.n_output)
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight.data)
#
#
#     def forward(self, x):
#         r, _ = self.SEN(x.transpose(1, 0))
#
#         r = r[-1, :, :]
#         r = torch.cat([r, x[:, -1, :]], 1)
#
#         y = self.KDN(r)
#
#         return y
#

def train():
    loader = DataLoader()
    model = MODEL(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LR)


    train_x, train_y = loader.getStandardTrainDataSet()

    train_x = torch.from_numpy(train_x).type(torch.float).to(DEVICE)
    train_y = torch.from_numpy(train_y).type(torch.float).to(DEVICE)

    train_dataset_loader = torchDataLoader(dataset=DiabetesDataset(train_x, train_y),
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           drop_last=False)

    for epoch in tqdm(range(N_EPOCHS)):
        for i, (x, y) in enumerate(train_dataset_loader, 0):
            model.train()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            # optimizer mode
            optimizer.step()

    path = os.path.join(
        os.getcwd(),
        'pre_train_model',
        'dfmnet_for_vis',
        'model.pt'
    )
    torch.save(model, path)

    with torch.no_grad():
        model.eval()
        for tag in loader.dataset_tags:
            print("Test: ", tag)
            x_data, y_data = loader.getStandardTestDataSet(tag)
            x_data_torch = torch.from_numpy(x_data).type(torch.float).to(DEVICE)
            pred = model(x_data_torch)
            print(tag, " : ", np.sqrt(mean_squared_error(y_data, pred.to('cpu').detach().numpy())))

if __name__ == '__main__':
    train()
