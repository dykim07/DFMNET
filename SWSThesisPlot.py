import pickle
import os

import numpy as np 



class SWSVisualizer():
    def __init__(self, tag:str):
        self.tag = tag
        self.loadResults(self.tag)

    def loadResults(self, tag:str):
        save_path = os.path.join(
            'pre_train_model',
            'dfmnet_for_vis',
            tag + '_sensor_test_result.pick'
        )

        with open(save_path, 'wb') as f:
            pickle.dump({
                'test_y' : test_y,
                'predictions': predictions,
                'MSEs': MSEs
        },f )