import errno
import os
import pickle
from sklearn.metrics import mean_squared_error as MSE
import numpy as np

def makeFolder(location):
    if not os.path.exists(location):
        try:
            os.makedirs(location)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return 0

def postProcessing(fileName:str):
    path = os.path.join(
        os.getcwd(),
        'Results',
        fileName
    )
    with open(path, 'rb') as f:
        results = pickle.load(f)
        for key, data in results.items():
            print(key, np.sqrt(MSE(data['y_data'], data['y_pred'])) )

