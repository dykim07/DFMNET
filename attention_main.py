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
#from tensorboardX import SummaryWriter

CUDA_ID = 2
N_EPOCHS = 30
BATCH_SIZE = 512

# times = strftime("%y%m%d_%H%M%S", localtime())
# SAVE_PATH = os.path.join(os.getcwd(), 'logdir')
# makeFolder(SAVE_PATH)

from Models.Atten import ATTENBase as MODEL

if torch.cuda.is_available():
    torch.cuda.set_device(CUDA_ID)
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

if __name__ == '__main__':
    loader = DataLoader()

    model = MODEL(batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, writer=None, device=DEVICE)
    model.to(DEVICE)
    
    train_x, train_y = loader.getStandardTrainDataSet()
    results = dict()

    for tag in loader.dataset_tags:
        print("Test: ", tag)
        x_data, y_data = loader.getStandardTestDataSet(tag)
        x_data_torch = torch.from_numpy(x_data).type(torch.float).to(DEVICE)
        
        results[tag] = {
            'x_data': x_data,
            'y_data': y_data,
            'y_pred': pred.to('cpu').detach().numpy()
        }

    with open(os.path.join(os.getcwd(), 'Results', 'atten_test.pick'), 'wb') as f:
        pickle.dump(results, f)


    # def train(self, X: np.ndarray, y: np.ndarray, X_valid=None, y_valid=None):

    #     # preprocessing
    #     self.train_x = torch.from_numpy(train_x).type(torch.float).to(self.device)
    #     self.train_y = torch.from_numpy(train_y).type(torch.float).to(self.device)

    #     train_dataset_loader = torchDataLoader(dataset=DiabetesDataset(self.train_x, self.train_y),
    #                                            batch_size=self.batch_size,
    #                                            shuffle=True,
    #                                            drop_last=False)

    #     if (X_valid is not None) and (y_valid is not None):
    #         self.valid_data = True
    #     else:
    #         pass

    #     for epoch in tqdm(range(self.n_epochs)):
    #         for i, (x, y) in enumerate(train_dataset_loader, 0):
    #             self.train()
    #             self.optimizer.zero_grad()
    #             y_pred = self(x)
    #             loss = self.criterion(y_pred, y)
    #             loss.backward()

    #             # optimizer mode
    #             if self.valid_data:
    #                 pass
    #             else:
    #                 self.optimizer.step()
    #         if self.writer is not None:
    #             self.writer.add_scalar('train/train_loss', loss.item(), epoch )

    #     return self

