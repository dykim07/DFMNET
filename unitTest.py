import numpy as np
from Utils.trainUtils import *
from dataset.DataLoader import DataLoader, DiabetesDataset

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_name = os.path.join(
    os.getcwd(),
    'Results',
    'test.pick'
)

with open(file_name, 'rb') as f:
    results = pickle.load(f)

gt_array = []
pred_array = []

for tag, item in results.items():
    print("Test: ", tag)
    y_data = item['y_data']
    y_pred = item['y_pred']
    print(np.sqrt(mean_squared_error(y_data, y_pred)))
    gt_array.append(y_data)
    pred_array.append(y_pred)

g = np.concatenate(gt_array)
p = np.concatenate(pred_array)
print(g.shape)
print(p.shape)
print(np.sqrt(mean_squared_error(g,p)))


