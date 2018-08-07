import numpy as np
from Utils.trainUtils import *
from dataset.DataLoader import DataLoader, DiabetesDataset

from sklearn.linear_model import LinearRegression
from sklearn import svm


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

INPUT_DIM = 20
OUTPUT_DIM = 39
CUDA_ID = 2
N_EPOCHS = 30
BATCH_SIZE = 512

loader = DataLoader()
train_x, train_y = loader.getStandardTrainDataSet()
poly = PolynomialFeatures(1)
model = svm.SVR()
print(train_x[:, -1, :].shape)
print(train_y.shape)
model.fit(train_x[:, -1, :], train_y)

gt_array = []
pred_array = []

for tag in loader.dataset_tags:
    print("Test: ", tag)
    x_data, y_data = loader.getStandardTestDataSet(tag)
    y_pred = model.predict(poly.fit_transform(x_data[:, -1, :]))
    print(np.sqrt(mean_squared_error(y_data, y_pred)))
    gt_array.append(y_data)
    pred_array.append(y_pred)

g = np.concatenate(gt_array)
p = np.concatenate(pred_array)
print(g.shape)
print(p.shape)
print(np.sqrt(mean_squared_error(g,p)))



