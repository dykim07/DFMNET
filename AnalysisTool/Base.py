import torch
from AnalysisTool import SWSVis

class AnalyzerBase():
    def __init__(self, ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def loadModel(self, model_path:str):
        self.model = torch.load(model_path).to(self.device)

    def setSWSVis(self):
        self.sws = SWSVis(self.model, self.joint_names)

    def init(self):
        self.loadDataSet()
        self.setSWSVis()