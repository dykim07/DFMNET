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
import matplotlib.pyplot as plt

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



class SWSVis():
    def __init__(self,
                 model_loc:str):
        self.model_loc = model_loc
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load model
        self.model = torch.load(self.model_loc).to(self.device)
        self.model.eval()

        # load dataset
        self.loadDataSet()
        self.criterion = nn.MSELoss()

        self.setBasePredictionResults()

        # 13 joints
        self.JointNames = [
            'Atlas',
            'RightBack',
            'RightShoulder',
            'RightElbow',
            'RightWrist',
            'LeftBack',
            'LeftShoulder',
            'LeftElbow',
            'LeftWrist',
            'RightKnee',
            'RightAnkle',
            'LeftKnee',
            'LeftAnkle'
        ]

        print("load...")

    def loadDataSet(self):
        # load dataset
        self.dataloader = DataLoader()
        self.train_x, self.train_y = self.dataloader.getStandardTrainDataSet()

        self.train_x = torch.from_numpy(self.train_x).type(torch.float).to(self.device)
        self.train_y = torch.from_numpy(self.train_y).type(torch.float).to(self.device)

        self.test_x = dict()
        self.test_y = dict()

        for tag in self.dataloader.getDataSetTags():
            test_x, test_y = self.dataloader.getStandardTestDataSet(tag)
            self.test_x[tag] = torch.from_numpy(test_x).type(torch.float).to(self.device)
            self.test_y[tag] = torch.from_numpy(test_y).type(torch.float).to(self.device)

    def predict(self, input):
        with torch.no_grad():
            self.model.eval()
            return self.model(input)

    def setBasePredictionResults(self):
        self.output_normal = dict()
        self.normal_loss = dict()

        for tag in self.dataloader.getDataSetTags():
            self.output_normal[tag] = self.predict(self.test_x[tag])
            self.normal_loss[tag] = self.criterion(self.output_normal[tag], self.test_y[tag])
            print(tag, self.normal_loss[tag])

    def zero_sensor_all(self, target_sensor_id, tag):
        output_zero = self.zero_sensor(target_sensor_id, tag)
        zero_loss = self.criterion(output_zero, self.test_y[tag])
        return zero_loss.item() - self.normal_loss[tag].item()

    def zero_sensor(self, target_sensor_id, tag):
        zero_input = self.test_x[tag].clone()
        zero_input[:, :, target_sensor_id] = 0.
        output_zero = self.predict(zero_input)
        return output_zero

    def zero_sensor_all_outputs(self):
        loss = dict()
        tags = self.dataloader.getDataSetTags()
        for tag in tags:
            loss[tag] = np.array([self.zero_sensor_all(idx, tag) for idx in range(self.test_x[tag].size(-1))])

        loss_relative = dict()
        all_loss = np.array([loss[tag] for tag in tags])
        all_loss = all_loss.sum(axis=0)
        loss_relative['all'] = all_loss / all_loss.sum()
        for tag in tags:
            loss_relative[tag] = loss[tag] / loss[tag].sum()
        return loss_relative

    def Tabular(self):
        loss_mat = dict()

        loss_tag = dict()
        for tag in self.dataloader.getDataSetTags():
            loss_tag[tag] = torch.zeros(len(self.JointNames), 20)
        for target_sensor_idx in range(20):
            for tag in self.dataloader.getDataSetTags():
                target = self.output_normal[tag].reshape(self.output_normal[tag].size(0), 13, 3)
                pred = self.zero_sensor(target_sensor_idx, tag).reshape(self.output_normal[tag].size(0), 13, 3)

def overall():
    path = os.path.join(os.getcwd(), 'pre_train_model', 'dfmnet_for_vis', 'model.pt')
    vis = SWSVis(path)
    loss = dict()

    tags = ['SQ', 'BR', 'WM']
    for tag in tags:
        loss[tag] = np.array([vis.zero_sensor_all(idx, tag) for idx in range(20)])

    index = np.arange(0, 20)
    plt.figure(figsize=(8,8))
    plt.subplot(221)
    # overall
    all_loss = np.array([loss[tag] for tag in tags ])
    all_loss = all_loss.sum(axis=0)
    all_loss = all_loss / all_loss.sum()
    plt.bar(index, all_loss)
    plt.xticks(index)
    plt.title('Ovarall')
    plt.grid()
    plt.xlabel('sensor index')
    plt.ylabel('ratio')

    for idx, tag in enumerate(tags):
        n_plot = '22' + str(idx+2)
        plt.subplot(n_plot)
        print(loss[tag])
        loss_v = loss[tag] / loss[tag].sum()
        plt.bar(index, loss_v)
        plt.xticks(index)
        plt.title(tag)
        plt.grid()
        plt.xlabel('sensor index')
        plt.ylabel('ratio')
    plt.tight_layout()
    plt.show()

def table():
    path = os.path.join(os.getcwd(), 'pre_train_model', 'dfmnet_for_vis', 'model.pt')
    vis = SWSVis(path)
    vis.Tabular()



if __name__ == '__main__':
    overall()





    # loss = loss / loss.sum()
    # index = np.arange(0, 20)
    #
    # plt.bar(index, loss)
    # plt.xticks(index)
    # plt.title(tag)
    # plt.grid()
    # plt.xlabel('sensor index')
    # plt.ylabel('ratio')
    # plt.show()
