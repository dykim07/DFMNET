import torch
import torch.nn as nn
import torch.optim as optim

class ATTENBase(nn.Module):
    def __init__(self,
    n_input = 20, 
    n_output = 39,
    n_gru_layer = 2,
    n_gru_hidden = 128,
    n_kdn_hidden = 128,
    batch_size = 512,
    writer = None,
    device = 'cuda'):

    super().__init__()

    self.n_input = n_input
    self.n_output = n_output
    self.n_gru_layer = n_gru_layer
    self.n_gru_hidden = n_gru_hidden
    self.n_kdn_hidden = n_kdn_hidden
    self.batch_size = batch_size
    self.writer = writer
    self.device = device 
    self._buildModel()

def _buildModel(self):
    self.endcoder = nn.GRU(
        input_size = self.n_input,
        hidden_size = self.n_hidden,
        num_layer = self.n_gru_layer,
        bidirectional = True,
        dropout = 0.5 
    )

    self.attention_layer = nn.Sequential(
        
    )




