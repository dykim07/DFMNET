import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
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

        with open(save_path, 'rb') as f:
            results = pickle.load(f)

        self.zero_predictions = results['zero_predictions']
        self.normal_predictions = results['normal_predictions']
        self.test_y = results['test_y']

        self.time_zero_mse = []
        for pred in self.zero_predictions:
            self.time_zero_mse.append(self.MSETime(pred, self.test_y))
        self.time_zero_mse = np.vstack(self.time_zero_mse).T
        self.time_normal_mse = self.MSETime(self.normal_predictions, self.test_y).reshape(-1, 1)

        self.gab = np.abs(self.time_normal_mse - self.time_zero_mse)

        self.index = np.arange(0, 20)  + 1
        self.n_frame = 1000
        self.initDraw()


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



    def initDraw(self):


        self.order = [0, 2, 1]
        self.skeleton_index_list = [
            [0, 1], # back bone
            [1, 2, 4 ,5, 6], # L arm 3 - 5
            [1, 3, 7, 8, 9],
            [0, 10, 11],
            [0, 12, 13]
        ]


        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(1, 2)

        # motion
        self.ax_motion = self.fig.add_subplot(self.gs[0, 0], projection='3d')
        motion = self.test_y.reshape(self.test_y.shape[0], -1, 3)[0, :, :]
        motion = np.concatenate([np.zeros([1, 3]), motion], axis=0)
        self.lines_motion = []

        for sk_idx in  self.skeleton_index_list:
            L = self.ax_motion.plot(motion[sk_idx, self.order[0]],
                               motion[sk_idx, self.order[1]],
                               motion[sk_idx, self.order[2]] )
            self.lines_motion.append(L)

        self.ax_motion.set_xlim3d(-0.3, 0.3)
        self.ax_motion.set_ylim3d(-1, 1)
        self.ax_motion.set_zlim3d(0, 1.5)
        self.ax_motion.set_aspect('equal')
        self.ax_motion.set_axis_off()
        self.ax_motion.view_init(10, 50)

        self.ax_motion.set_xticklabels([])
        self.ax_motion.set_yticklabels([])
        self.ax_motion.set_zticklabels([])

        self.ax_motion.grid(False)



        # sensor
        self.ax_sensor = self.fig.add_subplot(self.gs[0, 1])
        self.barchart = self.ax_sensor.bar(self.index, self.gab[0, :])
        self.ax_sensor.grid()
        self.ax_sensor.set_ylabel("Effect")
        self.ax_sensor.set_xlabel("Sensor ID")
        self.ax_sensor.set_ylim([0, 1])
        self.fig.tight_layout()
        
    def update(self, num):
        motion = self.test_y.reshape(self.test_y.shape[0], -1, 3)[num, :, :]
        motion = np.concatenate([np.zeros([1, 3]), motion], axis=0)
        for sk_index, motion_line in zip(self.skeleton_index_list, self.lines_motion):
            motion_line[0].set_data(
                motion[sk_index, self.order[0]],
                motion[sk_index, self.order[1]]
            )

            motion_line[0].set_3d_properties(
                motion[sk_index, self.order[2]]
            )

        # sensor
        mses = self.gab[num, :]
        mses = mses / np.sum(mses)
        for idx, chart in enumerate(self.barchart):
            chart.set_height(mses[idx])
  
    def show(self):
        ani = animation.FuncAnimation(self.fig, self.update, self.n_frame, interval=20, blit=False)
        plt.show()

    def save(self, is_slow=True):
        if is_slow:
            fps = 30
        else:
            fps = 120
        
        fileName = os.path.join(
            os.getcwd(),
            self.tag + '_eff.mp4'
        )

        writer = animation.writers['ffmpeg']
        writer = writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        line_ani = animation.FuncAnimation(self.fig, self.update, self.n_frame, interval=20, blit=False)
        line_ani.save(fileName, writer=writer)

if __name__ == '__main__':
    vis = SWSVisualizer('WM')
    vis.save()

    vis = SWSVisualizer('SQ')
    vis.save()
    
    vis = SWSVisualizer('BR')
    vis.save()
