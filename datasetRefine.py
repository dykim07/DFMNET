from copy import deepcopy
import os
import numpy as np
import quaternion
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class Viwer():
    def __init__(self, tag, dataset,  order = [0, 2, 1]):
        self.tag = tag
        self.order = order
        self.dataset = dataset.reshape(dataset.shape[0], 13, 3)
        self.fig = plt.figure(figsize=(5, 5))
        self.skeleton_index_list = [
            [0, 1],  # back bone
            [1, 2, 4, 5, 6],  # L arm 3 - 5
            [1, 3, 7, 8, 9],
            [0, 10, 11],
            [0, 12, 13]
        ]

        self.initDraw()

    def initDraw(self):
        self.axMotion = self.fig.add_subplot(111, projection='3d')
        motion = self.dataset[0, :, :]
        motion = np.concatenate([np.zeros([1, 3]), motion], axis=0)

        self.lines_motion = []
        for sk_idx in  self.skeleton_index_list:
            L = self.axMotion.plot(motion[sk_idx, self.order[0]],
                               motion[sk_idx, self.order[1]],
                               motion[sk_idx, self.order[2]] )
            self.lines_motion.append(L)

        self.axMotion.set_xlim3d(-0.3, 0.3)
        self.axMotion.set_ylim3d(-1, 1)
        self.axMotion.set_zlim3d(0, 2)
        self.axMotion.set_aspect('equal')

        # self.axMotion.set_axis_off()
        # self.axMotion.view_init(10, 50)
        #
        # self.axMotion.set_xticklabels([])
        # self.axMotion.set_yticklabels([])
        # self.axMotion.set_zticklabels([])
        # self.axMotion.grid(False)

    def update(self, num):
        motion = self.dataset[num, :, :]
        motion = np.concatenate([np.zeros([1, 3]), motion], axis=0)

        for sk_index, motion_line in zip(self.skeleton_index_list, self.lines_motion):
            motion_line[0].set_data(
                motion[sk_index, self.order[0]],
                motion[sk_index, self.order[1]]
            )

            motion_line[0].set_3d_properties(
                motion[sk_index, self.order[2]]
            )

    def show(self):
        line_ani = animation.FuncAnimation(self.fig, self.update, self.dataset.shape[0],
                                           interval=int(1000 / 120), blit=False)
        plt.show()



class DataLoader():
    def __init__(self, window_size=120):
        self.window_size = window_size
        self.dataset_tags = ['BR', 'SQ', 'WM']
        self.file_names = ['BnR.mat', 'squat.mat', 'windmill.mat']
        self.dataset = dict()
        self.scaler = None
        self.load_dataset()

    def load_dataset(self):
        for idx, tag in enumerate(self.dataset_tags):
            file_name = os.getcwd() + '/dataset/' + self.file_names[idx]
            print(file_name)
            self.dataset[tag] = loadmat(file_name)

    def forwardKinematics(self, y_data, tag):
        hip_data = self.dataset[tag]['test_hip']
        hip_quart = quaternion.as_quat_array(hip_data[:, :4].copy())
        hip_position = hip_data[:, 4:].copy()

        y_data = deepcopy(y_data)
        outs = np.zeros_like(y_data)

        marker_idxs = np.array(range(y_data.shape[1])).reshape(-1, 3)

        for idx in range(y_data.shape[0]):
            rtMat = quaternion.as_rotation_matrix(hip_quart[idx])
            pos = hip_position[idx]
            for marker_idx in marker_idxs:
                outs[idx, marker_idx] = np.matmul(rtMat.T, (y_data[idx, marker_idx])) + pos

        return outs
def refine():
    tag = 'SQ'
    loader = DataLoader()
    test_y = loader.dataset[tag]['test_y']
    test_hip = loader.dataset[tag]['test_hip']
    print(test_y.shape)
    print(test_hip.shape)

    fk_y = loader.forwardKinematics(test_y, tag)
    viwer = Viwer(tag = tag, dataset=fk_y)
    viwer.show()

if __name__ == '__main__':
    refine()
