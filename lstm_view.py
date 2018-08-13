import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import multiprocessing
from multiprocessing import Pool


from mpl_toolkits.mplot3d import Axes3D
import pickle

from Utils.trainUtils import *
from dataset.DataLoader import DataLoader, DiabetesDataset
from tensorboardX import SummaryWriter
from scipy.signal import medfilt, savgol_filter

# MODELs

INPUT_DIM = 20
OUTPUT_DIM = 39
CUDA_ID = 0
N_EPOCHS = 30
BATCH_SIZE = 512
LR = 0.001

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

class DFMNET(nn.Module, BaseEstimator, RegressorMixin):
    def __init__(self, n_input: int, n_output: int, n_lstm_layer=2, n_lstm_hidden=128, n_KDN_hidden=128, device='cuda'):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_lstm_layer = n_lstm_layer
        self.n_lstm_hidden = n_lstm_hidden
        self.n_KDN_hidden = n_KDN_hidden
        self.device = device
        self._build_model()

    def _build_model(self):

        self.SEN = nn.LSTM(
            input_size=self.n_input,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.n_lstm_layer,
            dropout=0.5
        )

        self.KDN = nn.Sequential(
            nn.Linear(self.n_lstm_hidden + self.n_input, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_output)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)


    def forward(self, x):
        r, _ = self.SEN(x.transpose(1, 0))

        r = r[-1, :, :]
        r = torch.cat([r, x[:, -1, :]], 1)

        y = self.KDN(r)

        return y


class DFMNET_NA(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_lstm_layer=2, n_lstm_hidden=128, n_KDN_hidden=128, device='cuda'):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_lstm_layer = n_lstm_layer
        self.n_lstm_hidden = n_lstm_hidden
        self.n_KDN_hidden = n_KDN_hidden
        self.device = device
        self._build_model()

    def _build_model(self):

        self.SEN = nn.LSTM(
            input_size=self.n_input,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.n_lstm_layer,
            dropout=0.5
        )

        self.KDN = nn.Sequential(
            nn.Linear(self.n_lstm_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_KDN_hidden),
            nn.ReLU(),

            nn.Linear(self.n_KDN_hidden, self.n_output)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)


    def forward(self, x):
        r, _ = self.SEN(x.transpose(1, 0))
        r = r[-1, :, :]
        y = self.KDN(r)

        return y


def training():
    loader = DataLoader()
    writer = SummaryWriter()

    model = DFMNET_NA(INPUT_DIM, OUTPUT_DIM)
    model = model.to(DEVICE)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_x, train_y = loader.getStandardTrainDataSet()

    train_x = torch.from_numpy(train_x).type(torch.float).to(DEVICE)
    train_y = torch.from_numpy(train_y).type(torch.float).to(DEVICE)

    train_dataset_loader = torchDataLoader(dataset=DiabetesDataset(train_x, train_y),
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           drop_last=False)

    for epoch in tqdm(range(N_EPOCHS)):
        for i, (x, y) in enumerate(train_dataset_loader, 0):
            model.train()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        writer.add_scalar('train/loss', loss.item(), epoch)

    writer.close()

    path = os.path.join(
        os.getcwd(),
        'pre_train_model',
        'dfmnet_no_atten.pt'
    )
    torch.save(model, path)








def analysis():
    path = os.path.join(
        os.getcwd(),
        'pre_train_model',
        'dfmnet_na',
        'model.pt'
    )
    model = torch.load(path).to('cpu').eval()
    params = model.SEN.state_dict()
    print(params.keys())

    # ----------------
    # load params
    # ----------------

    w_ii0, w_if0, w_ig0, w_io0 = parseWeights(params['weight_ih_l0'], 128, 20)
    w_hi0, w_hf0, w_hg0, w_ho0 = parseWeights(params['weight_hh_l0'], 128, 128)
    b_ii0, b_if0, b_ig0, b_io0 = parsebias(params['bias_ih_l0'], 128)
    b_hi0, b_hf0, b_hg0, b_ho0 = parsebias(params['bias_hh_l0'], 128)

    w_ii1, w_if1, w_ig1, w_io1 = parseWeights(params['weight_ih_l1'], 128, 128)
    w_hi1, w_hf1, w_hg1, w_ho1 = parseWeights(params['weight_hh_l1'], 128, 128)

    b_ii1, b_if1, b_ig1, b_io1 = parsebias(params['bias_ih_l1'], 128)
    b_hi1, b_hf1, b_hg1, b_ho1 = parsebias(params['bias_hh_l1'], 128)

    # ----------------
    # load params
    # ----------------

    loader = DataLoader()
    writer = SummaryWriter()
    tags = loader.dataset_tags


    train_x, train_y = loader.getStandardTrainDataSet()

    for tag in tags:
        x_data, y_data = loader.getStandardTestDataSet(tag)
        x_data = torch.from_numpy(x_data).type(torch.float)
        N = 0

        forget_0 = []
        forget_1 = []
        output_0 = []
        output_1 = []
        input_0 = []
        input_1 = []

        for idx_dataset in tqdm(range(x_data.size(0))):

            ft0_array = []
            ot0_array = []
            it0_array = []

            ft1_array = []
            ot1_array = []
            it1_array = []

            for idx_seq in range(x_data.size(1)):
                # loop start
                h0 = torch.zeros(128)
                c0 = torch.zeros(128)

                h1 = torch.zeros(128)
                c1 = torch.zeros(128)
                # first layer
                x = x_data[idx_dataset, idx_seq, :]
                it0 = inputGate(w_ii0, b_ii0, w_hi0, b_hi0, h0, x)
                ft0 = forgetGate(w_if0, b_if0, w_hf0, b_hf0, h0,  x)
                gt0 = cellState(w_ig0, b_ig0, w_hg0, b_hg0, h0, x)
                ot0 = outputGate(w_io0, b_io0, w_ho0, b_ho0, h0, x)
                c0 = nextCell(ft0, c0, it0, gt0)
                h0 = nextHidden(ot0, c0)

                # second_layer
                it1 = inputGate(w_ii1, b_ii1, w_hi1, b_hi1, h1, h0)
                ft1 = forgetGate(w_if1, b_if1, w_hf1, b_hf1, h1, h0)
                gt1 = cellState(w_ig1, b_ig1, w_hg1, b_hg1, h1, h0)
                ot1 = outputGate(w_io1, b_io1, w_ho1, b_ho1, h1, h0)
                c1 = nextCell(ft1, c1, it1, gt1)
                h1 = nextHidden(ot1, c1)

                ft0_array.append(ft0.reshape(1,-1))
                ot0_array.append(ot0.reshape(1, -1))
                it0_array.append(it0.reshape(1, -1))

                ft1_array.append(ft1.reshape(1,-1))
                ot1_array.append(ot1.reshape(1, -1))
                it1_array.append(it1.reshape(1, -1))

            meanFt0 = np.mean(torch.cat(ft0_array).numpy(), axis=1)
            meanot0 = np.mean(torch.cat(ot0_array).numpy(), axis=1)
            meanit0 = np.mean(torch.cat(it0_array).numpy(), axis=1)

            meanFt1 = np.mean(torch.cat(ft1_array).numpy(), axis=1)
            meanot1 = np.mean(torch.cat(ot1_array).numpy(), axis=1)
            meanit1 = np.mean(torch.cat(it1_array).numpy(), axis=1)

            forget_0.append(meanFt0)
            forget_1.append(meanFt1)

            output_0.append(meanot0)
            output_1.append(meanot1)

            input_0.append(meanit0)
            input_1.append(meanit1)

        path = os.path.join(os.getcwd(), 'pre_train_model', 'dfmnet_na','lstm_view_' + str(tag) + '.pick' )
        with open(path, 'wb') as f:
            pickle.dump(
                {
                    'f0': forget_0,
                    'f1': forget_1,
                    'o0': output_0,
                    'o1': output_1,
                    'i0': input_0,
                    'i1': input_1
                }, f
            )



    # plt.figure()
    # plt.plot(meanFt0)
    # plt.plot(meanFt1)
    # plt.plot(meanot0)
    # plt.plot(meanot1)
    # plt.legend(['FT0', 'FT1', 'OT0', 'OT1'])
    # plt.show()


def parseWeights(params, hs0, hs1):
    params = params.reshape(4, hs0, hs1)
    return [params[i, :, :] for i in range(4)]

def parsebias(params, hs0):
    params = params.reshape(4, hs0)
    return [params[i, :] for i in range(4)]

# lstm funtions
def inputGate(w_ii, b_ii, w_hi, b_hi, h, x):
    return torch.sigmoid(w_ii @ x + b_ii + w_hi @ h + b_hi)

def forgetGate(w_if, b_if, w_hf, b_hf, h, x):
    return torch.sigmoid(w_if @ x + b_if + w_hf @ h + b_hf)

def cellState(w_ig, b_ig, w_hg, b_hg, h, x):
    return torch.tanh(w_ig @ x + b_ig + w_hg @ h + b_hg)

def outputGate(w_io, b_io, w_ho, b_ho, h, x):
    return torch.sigmoid(w_io @ x + b_io + w_ho @ h + b_ho)

def nextCell(f,c,i,g):
    return f* c + i * g

def nextHidden(o, c):
    return o * torch.tanh(c)



class WeightViwer():
    def __init__(self, base , tag='WM', order = [0, 2, 1]):
        self.tag = tag
        self.base = base
        self.order = order
        self.skeleton_index_list = [
            [0, 1], # back bone
            [1, 2, 4 ,5, 6], # L arm 3 - 5
            [1, 3, 7, 8, 9],
            [0, 10, 11],
            [0, 12, 13]
        ]


        self.loadDataset()
        self.fig = plt.figure(figsize=(8,5))
        self.gs = gridspec.GridSpec(2, 3)
        self.initDraw()
        self.fig.tight_layout()



    def loadDataset(self):
        self.loader = DataLoader()
        writer = SummaryWriter()
        tags = self.loader.dataset_tags
        train_x, train_y = self.loader.getStandardTrainDataSet()
        self.test_x, self.test_y = self.loader.getTestDataSet(self.tag)

        print(self.test_y.shape)
        self.grad = np.mean(np.absolute(np.gradient(self.test_y, axis=0)), axis=1) * 120

        print('min: ',  np.min(self.grad))
        print('max: ', np.max(self.grad))

        path = os.path.join(
            os.getcwd(),
            'pre_train_model',
            self.base,
            'lstm_view_' + self.tag + '.pick'
        )
        with open(path, 'rb') as f:
            self.weights_dict = pickle.load(f)

        self.positions = self.test_y.reshape(self.test_y.shape[0], -1, 3)

    def initDraw(self):

        self.axSpeed = self.fig.add_subplot(self.gs[:, 0])
        self.bar =self.axSpeed.bar([0], self.grad[0])
        self.axSpeed.set_ylim(0, 1)
        self.axSpeed.set_xticks([0])
        self.axSpeed.set_xlabel('Speed')
        self.axSpeed.set_ylabel('m/s')
        self.axSpeed.set_title("Motion Velocity")

        self.axCell0 = self.fig.add_subplot(self.gs[0, 1])
        self.axCell0.set_ylim(0.4, 0.55)
        self.line_f0 = self.axCell0.plot(self.weights_dict['f0'][0])
        self.line_o0 = self.axCell0.plot(self.weights_dict['o0'][0])
        self.line_i0 = self.axCell0.plot(self.weights_dict['i0'][0])
        self.axCell0.set_title("LSMT 0")

        self.axCell1 = self.fig.add_subplot(self.gs[1, 1])
        self.axCell1.set_ylim(0.4, 0.55)
        self.line_f1 = self.axCell1.plot(self.weights_dict['f1'][0])
        self.line_o1 = self.axCell1.plot(self.weights_dict['o1'][0])
        self.line_i1 = self.axCell1.plot(self.weights_dict['i1'][0])

        self.axCell1.legend(
            ['forget', 'output', 'input']
        )
        self.axCell1.set_title("LSMT 1")

        self.axMotion = self.fig.add_subplot(self.gs[:, 2], projection='3d')


        motion = self.positions[0, :, :]
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
        self.axMotion.set_axis_off()
        self.axMotion.view_init(10, 50)

        self.axMotion.set_xticklabels([])
        self.axMotion.set_yticklabels([])
        self.axMotion.set_zticklabels([])

        # self.axMotion.set_zticks([])
        #
        self.axMotion.grid(False)
        #
        # self.axMotion.xaxis.pane.set_edgecolor('white')
        # self.axMotion.yaxis.pane.set_edgecolor('white')
        #
        # self.axMotion.xaxis.pane.fill = False
        # self.axMotion.yaxis.pane.fill = False
        # self.axMotion.zaxis.pane.fill = True
        #
        # self.axMotion.w_zaxis.line.set_lw(0.0)

    def update(self, num):
        self.bar[0].set_height(self.grad[num])
        self.line_f0[0].set_ydata(self.weights_dict['f0'][num])
        self.line_o0[0].set_ydata(self.weights_dict['o0'][num])
        self.line_i0[0].set_ydata(self.weights_dict['i0'][num])

        self.line_f1[0].set_ydata(self.weights_dict['f1'][num])
        self.line_o1[0].set_ydata(self.weights_dict['o1'][num])
        self.line_i1[0].set_ydata(self.weights_dict['i1'][num])


        motion = self.positions[num, :, :]
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
        line_ani = animation.FuncAnimation(self.fig, self.update, len(self.weights_dict['f0']),
                                           interval=int(1000 / 120), blit=False)
        plt.show()

    def save(self):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=1800)
        line_ani = animation.FuncAnimation(self.fig, self.update, len(self.weights_dict['f0']),
                                           interval=int(1000 / 120), blit=False)
        line_ani.save(self.tag + '.mp4', writer=writer)



def performaceEval(base = 'dfmnet_na'):
    path = os.path.join(
        os.getcwd(),
        'pre_train_model',
        base,
        'model.pt'
    )

    model = torch.load(path).to('cuda')
    # dataset =
    # results = []
    # for tag in



def multiwriter(key):
    viwer = WeightViwer('dfmnet_na', key)
    viwer.save()


if __name__ == '__main__':
    n_core = multiprocessing.cpu_count() - 1
    print("Start with ", n_core, " CPUs")
    with Pool(n_core) as p:
        r = list(p.imap(multiwriter, ["BR", "WM", "SQ"]))

    # viwer = WeightViwer("WM")
    # viwer.show()

    #    training()
    #    analysis()
    # for key in ["BR", "WM", "SQ"]:
    #     viwer = WeightViwer('dfmnet_na', key)
    #     viwer.save()

#    viwer.show()
