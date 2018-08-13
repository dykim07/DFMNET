import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin


import pickle

from Utils.trainUtils import *
from dataset.DataLoader import DataLoader, DiabetesDataset
from tensorboardX import SummaryWriter


# MODELs

INPUT_DIM = 20
OUTPUT_DIM = 39
CUDA_ID = 0
N_EPOCHS = 30
BATCH_SIZE = 512
LR = 0.001

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

class DFMNET(nn.Module, BaseEstimator, RegressorMixin):
    def __init__(self, n_input: int, n_output: int, n_lstm_layer=2, n_lstm_hidden=128, n_KDN_hidden=128, device='cuda'):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_lstm_layer = n_lstm_layer
        self.n_lstm_hidden = n_lstm_hidden
        self.n_KDN_hidden = n_KDN_hidden
        self.device = device
        self._build_model()

    def _build_model(self):
        self.SEN1 = nn.LSTMCell(
            input_size=self.n_input,
            hidden_size=self.n_lstm_hidden,
        )
        self.SENDrop = nn.Dropout()
        self.SEN2 = nn.LSTMCell(
            input_size=self.n_lstm_hidden,
            hidden_size=self.n_lstm_hidden,
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
        # x -> batch, seq, dim
        h1 = torch.zeros(x.size(0), self.n_lstm_hidden).type(torch.float).to(self.device)
        c1 = torch.zeros(x.size(0), self.n_lstm_hidden).type(torch.float).to(self.device)

        h2 = torch.zeros(x.size(0), self.n_lstm_hidden).type(torch.float).to(self.device)
        c2 = torch.zeros(x.size(0), self.n_lstm_hidden).type(torch.float).to(self.device)

        for idx_seq in range(x.size(1)):
            x_t = x[:, idx_seq, :]
            h1, c1 = self.SEN1(x_t, (h1, c1))
            h2, c2 = self.SEN2(self.SENDrop(h1), (h2, c2))

        r = torch.cat([h2, x[:, -1, :]], 1)
        y = self.KDN(r)

        return y, h2, r


def training():
    loader = DataLoader()
    writer = SummaryWriter()

    model = DFMNET(INPUT_DIM, OUTPUT_DIM)
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
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
            y_pred, h2, r = model(x)
            loss = criterion(y_pred, y)
            loss.backward()

        writer.add_scalar('train/loss', loss.item(), epoch)
        writer.add_histogram('train/h2', h2.to('cpu').detach().numpy(), epoch)
        writer.add_histogram('train/r', r.to('cpu').detach().numpy(), epoch)

    writer.close()

    path = os.path.join(
        os.getcwd(),
        'pre_train_model',
        'model.pt'
    )
    torch.save(model, path)

if __name__ == '__main__':
    training()

    # for tag in loader.dataset_tags:
    #     print("Test: ", tag)
    #     x_data, y_data = loader.getStandardTestDataSet(tag)
    #     x_data_torch = torch.from_numpy(x_data).type(torch.float).to(DEVICE)
    #     pred = dfmnet.predict(x_data_torch)
    #
    #     results[tag] = {
    #         'x_data': x_data,
    #         'y_data': y_data,
    #         'y_pred': pred.to('cpu').detach().numpy()
    #     }
    #
    # with open(os.path.join(os.getcwd(), 'Results', 'tt_test.pick'), 'wb') as f:
    #     pickle.dump(results, f)
    #
    #
    # #writer.close()