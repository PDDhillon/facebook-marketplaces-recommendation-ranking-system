import torch
class FBMClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # load pre trained resnet50 from torch hub
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # change final layer of resnet-50 to have an output size = number of possible categories (13 categories)
        self.resnet50.fc = torch.nn.Linear(2048,13)

    def forward(self, X):
        return self.resnet50(X)