import torch
import torch.nn as nn
import numpy as np

class SWSVis():
    def __init__(self, model:torch.nn.Module, joint_names:list):
        self.model = model
        self.joint_names = joint_names
        self.criterion = nn.MSELoss()

    def predict(self, data):
        with torch.no_grad():
            self.model.eval()
            return self.model(data)

    def insertDataSet(self, input, outputGT):
        self.input = input
        self.output_GT = outputGT
        self.output_target = self.predict(self.input)
        self.generateBasePredictions()

    # zero sensor
    def zero_sensor_output(self, target_sensor_index):
        zero_input = self.input.clone()
        zero_input[:, :, target_sensor_index] = 0.
        output_zero = self.predict(zero_input)
        return output_zero

    def generateBasePredictions(self):
        self.base_predictions = [self.zero_sensor_output(idx) for idx in range(self.input.size(-1))]

    def relative_loss_for_all(self):
        loss_target = self.criterion(self.output_target, self.output_GT).item()
        loss_pred = np.array([self.criterion(pred, self.output_GT).item() for pred in self.base_predictions])
        return  loss_pred - loss_target

    def relative_loss_per_joint(self):
        criterion = nn.MSELoss(reduction='none')
        mse_mat = torch.zeros(self.input.size(-1), len(self.joint_names))
        target = self.output_target.reshape(-1, len(self.joint_names), 3)
        for idx_sensor in range(self.input.size(-1)):
            loss = criterion(self.base_predictions[idx_sensor].reshape(self.input.size(0), len(self.joint_names), 3), target)
            mse_mat[idx_sensor, :] = loss.mean(dim=2).mean(dim=0)
        return mse_mat.to('cpu').detach().numpy()