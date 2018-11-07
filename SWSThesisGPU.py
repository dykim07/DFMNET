import pickle
import os
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default = 1)
args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.cuda.set_device(args.cuda_id)

else:
    DEVICE = 'cpu'

class DFMNETSWS():
    def __init__(self):
        self.n_sensors = 20
        self.joint_names = [
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

        self.loadDataSet()
        self.loadModel()

    def loadDataSet(self):
        self.dataloader = DataLoader()
        train_x, train_y = self.dataloader.getStandardTrainDataSet()

        # for tag in self.dataloader.getDataSetTags():
        #     test_x, test_y = self.dataloader.getStandardTestDataSet(tag)
        #     self.test_x[tag] = torch.from_numpy(test_x).type(torch.float).to(DEVICE)
        #     self.test_y[tag] = torch.from_numpy(test_y).type(torch.float).to(DEVICE)

    def loadModel(self):
        save_path = os.path.join(
            os.getcwd(),
            'pre_train_model',
            'dfmnet_for_vis',
            'model.pt'
        )
        self.model =  torch.load(save_path, map_location=lambda storage, loc: storage)
        self.model = self.model.to(DEVICE)

    def testTag(self, tag:str):
        test_x, test_y = self.dataloader.getStandardTestDataSet(tag)
        test_x_torch = torch.from_numpy(test_x).type(torch.float).to(DEVICE)

        zero_predictions = []
        
        for sensor_idx in range(self.n_sensors):
            zero_predictions.append(
                self.prediction_of_zero_sensor(test_x_torch, sensor_idx)
            )
        
        normal_prediction = self.predict(test_x_torch).detach().to('cpu').numpy()
        
        save_path = os.path.join(
            'pre_train_model',
            'dfmnet_for_vis',
            tag + '_sensor_test_result.pick'
        )

        with open(save_path, 'wb') as f:
            pickle.dump({
                'test_y' : test_y,
                'zero_predictions': zero_predictions,
                'normal_predictions': normal_prediction,
            },f )

    def MSETime(self, pred:np.ndarray, target:np.ndarray ):
        """
        dataset shape : N by Dim 
        output = N by MSE
        """
        assert pred.shape == target.shape

        r = pred - target
        r = np.power(r, 2)
        r = np.mean(r, axis=1)
        return r

    def prediction_of_zero_sensor(self, input:torch.tensor, target_sensor_id):
        zero_input = input.clone()
        zero_input[:, :, target_sensor_id] = 0.
        return self.predict(zero_input).detach().to('cpu').numpy()

    def predict(self, input):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input)
        return pred

if __name__ == '__main__':
    sws = DFMNETSWS()
    sws.testTag('WM')
    sws.testTag('BR')    
    sws.testTag('SQ')


