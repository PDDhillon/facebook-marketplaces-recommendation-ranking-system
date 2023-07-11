import torch
class FBMClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # load pre trained resnet50 from torch hub
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # unfreeze the last two layers and retrain entire model   
        for param in self.resnet50.parameters():
            param.requires_grad = False
        # change final layer of resnet-50 to have an output size = number of possible categories (13 categories)
        self.resnet50.fc = torch.nn.Sequential(torch.nn.Linear(2048,100),torch.nn.ReLU(),torch.nn.Linear(100,13))

    def forward(self, X):
        return self.resnet50(X)
    
test = FBMClassifier()