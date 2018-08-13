# import numpy as np
# from Utils.trainUtils import *
# from dataset.DataLoader import DataLoader, DiabetesDataset
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

from dataset.dataset_old.DataLoader_older import DataLoader


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

def raw_loader_test():
    loader = DataLoader(window_size=120)
    (train_x, train_y), (test_x, test_y) = loader.splitDataSet2(loader.dataset_dic)

    for key, item in train_y.items():
        print(key)
        print(item.shape)

    for key, item in test_y.items():
        print(key)
        print(item.shape)








if __name__ == '__main__':
    loader = raw_loader_test()
