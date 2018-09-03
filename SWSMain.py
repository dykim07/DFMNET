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
from AnalysisTool.Base import AnalyzerBase

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
        results =dict()
        for tag in self.dataLoader.getDataSetTags():
            k = self.sws[tag].relative_loss_per_joint()
            results[tag] = k
        return results

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
        loss_v = results[tag]
        loss_v = sws[list(sws.keys())[0]].OverallScaler(loss_v)
        plt.bar(index, loss_v)
        plt.xticks(index)
        plt.title(tag)
        plt.grid()
        plt.xlabel('sensor index')
        plt.ylabel('ratio')
    plt.tight_layout()
    plt.show()

def plotJointEffectForDFMNET(ana, results, joint_names):
    plt.figure()

    overAll = np.zeros_like(results[ana.dataLoader.getDataSetTags()[0]])
    print(overAll.shape)
    for tag, plt_tag in zip(ana.dataLoader.getDataSetTags(),['222', '223', '224']) :
        overAll = overAll + results[tag]
        plt.subplot(plt_tag)
        plt.title(tag)
        plt.imshow(ana.sws[tag].JointScaler(results[tag]), cmap=plt.cm.Blues, interpolation='nearest')
        plt.xticks(np.arange(overAll.shape[1]), np.arange(overAll.shape[1]))
        plt.yticks(np.arange(len(joint_names)), joint_names)
        plt.colorbar()

    plt.subplot(221)
    plt.title("Overall")
    plt.imshow(ana.sws[tag].JointScaler(overAll), cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(overAll.shape[1]), np.arange(overAll.shape[1]))
    plt.yticks(np.arange(len(joint_names)), joint_names)
    plt.show()

def plotJointEffectForDFMNETUpperBody(ana, results, joint_names):
    plt.figure()
    overAll = np.zeros_like(results[ana.dataLoader.getDataSetTags()[0]][:9, :14])
    print(overAll.shape)
    for tag, plt_tag in zip(ana.dataLoader.getDataSetTags(),['222', '223', '224']) :
        overAll = overAll + results[tag][:9, :14]
        plt.subplot(plt_tag)
        plt.title(tag)
        plt.imshow(ana.sws[tag].JointScaler(results[tag][:9, :14]), cmap=plt.cm.Blues, interpolation='nearest')
        plt.xticks(np.arange(14), np.arange(14))
        plt.yticks(np.arange(9), joint_names[:9])
        plt.colorbar()

    plt.subplot(221)
    plt.title("Overall")
    plt.imshow(ana.sws[tag].JointScaler(overAll), cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(14), np.arange(14))
    plt.yticks(np.arange(9), joint_names[:9])
    plt.show()

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'pre_train_model', 'dfmnet_for_vis', 'model.pt')
    ana = DFMNETAnalyzer()
    ana.loadModel(path)
    ana.init()
    results = ana.JointView()
    plotJointEffectForDFMNETUpperBody(ana, results, ana.joint_names)



    # results = ana.OverAllView()
    # plotOverallBarChartForFMNET(results, ana.sws)
    # results = ana.JointView()

    # results = ana.JointView()
    # print(results)

    # for k, item in results.items():
    #     np.savetxt(k + '.csv', item, delimiter=',')
    # plt.figure()
    # plt.imshow(results, cmap=plt.cm.Blues, interpolation='nearest')
    # plt.colorbar()
    # plt.xticks(np.arange(20), np.arange(20))
    # plt.yticks(np.arange(len(ana.joint_names)), ana.joint_names)
    # plt.show()