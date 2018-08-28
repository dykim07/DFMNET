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

from AnalysisTool.SWSVis import SWSVis

class AnalyzerBase():
    def __init__(self, ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def loadModel(self, model_path:str):
        self.model = torch.load(model_path).to(self.device)

    def setSWSVis(self):
        self.sws = SWSVis(self.model, self.joint_names)

    def init(self):
        self.loadDataSet()
        self.setSWSVis()

class DFMNETAnalyzer(AnalyzerBase):
    def __init__(self):
        super(DFMNETAnalyzer, self).__init__()
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

    def loadDataSet(self):
        self.dataLoader = DataLoader()
        self.train_x, self.train_y = self.dataLoader.getStandardTrainDataSet()

        self.test_x_dict = dict()
        self.test_y_dict = dict()

        for tag in self.dataLoader.getDataSetTags():
            test_x, test_y = self.dataLoader.getStandardTestDataSet(tag)
            self.test_x_dict[tag] = torch.from_numpy(test_x).type(torch.float).to(self.device)
            self.test_y_dict[tag] = torch.from_numpy(test_y).type(torch.float).to(self.device)


    def setSWSVis(self):
        self.sws = dict()
        for tag in self.dataLoader.getDataSetTags():
            self.sws[tag] = SWSVis(self.model, self.joint_names)
            self.sws[tag].insertDataSet(self.test_x_dict[tag], self.test_y_dict[tag])

    def OverAllView(self):
        results = dict()
        for tag in self.dataLoader.getDataSetTags():
            k = self.sws[tag].relative_loss_for_all()
            results[tag] = k
        return results

    def JointView(self):
        tag = 'WM'
        return self.sws[tag].relative_loss_per_joint()

def plotOverallBarChartForFMNET(results:dict, sws:SWSVis):
    index = np.arange(0, 20)
    plt.figure(figsize=(8,8))
    plt.subplot(221)
    # overall
    all_loss = np.array([results[tag] for tag in results.keys()])
    all_loss = all_loss.sum(axis=0)
    all_loss = sws[list(sws.keys())[0]].OverallScaler(all_loss)
    plt.bar(index, all_loss)
    plt.xticks(index)
    plt.title('Ovarall')
    plt.grid()
    plt.xlabel('sensor index')
    plt.ylabel('ratio')

    for idx, tag in enumerate(results.keys()):
        n_plot = '22' + str(idx+2)
        plt.subplot(n_plot)
        loss_v = sws[tag].OverallScaler(results[tag])
        plt.bar(index, loss_v)
        plt.xticks(index)
        plt.title(tag)
        plt.grid()
        plt.xlabel('sensor index')
        plt.ylabel('ratio')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'pre_train_model', 'dfmnet_for_vis', 'model.pt')
    ana = DFMNETAnalyzer()
    ana.loadModel(path)
    ana.init()

    results = ana.OverAllView()
    plotOverallBarChartForFMNET(results, ana.sws)

    # results = ana.JointView()
    #
    # plt.figure()
    # plt.imshow(results, cmap=plt.cm.Blues, interpolation='nearest')
    # plt.colorbar()
    # plt.xticks(np.arange(20), np.arange(20))
    # plt.yticks(np.arange(len(ana.joint_names)), ana.joint_names)
    # plt.show()