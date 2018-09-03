import numpy as np
from Utils.trainUtils import *
from dataset.DataLoader import DataLoader, DiabetesDataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#
# from dataset.dataset_old.DataLoader_older import DataLoader
#
#
# def loader_v1():
#     file_name = os.path.join(
#         os.getcwd(),
#         'Results',
#         'test.pick'
#     )
#
#     with open(file_name, 'rb') as f:
#         results = pickle.load(f)
#
#     gt_array = []
#     pred_array = []
#
#     for tag, item in results.items():
#         print("Test: ", tag)
#         y_data = item['y_data']
#         y_pred = item['y_pred']
#         print(np.sqrt(mean_squared_error(y_data, y_pred)))
#         gt_array.append(y_data)
#         pred_array.append(y_pred)
#
#     g = np.concatenate(gt_array)
#     p = np.concatenate(pred_array)
#     print(g.shape)
#     print(p.shape)
#     print(np.sqrt(mean_squared_error(g,p)))
#
# def raw_loader_test():
#     loader = DataLoader(window_size=120)
#     (train_x, train_y), (test_x, test_y) = loader.splitDataSet2(loader.dataset_dic)
#
#     for key, item in train_y.items():
#         print(key)
#         print(item.shape)
#
#     for key, item in test_y.items():
#         print(key)
#         print(item.shape)

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




if __name__ == '__main__':
    JointMotionTest()
    path = os.path.join(os.getcwd(), 'pre_train_model', 'dfmnet_for_vis', 'model.pt')
    ana = DFMNETAnalyzer()
    ana.loadModel(path)
    ana.init()
