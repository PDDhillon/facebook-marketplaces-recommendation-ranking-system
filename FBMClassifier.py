import torch
class FBMClassifier(torch.nn.Module):
    '''A class to represent a classifier model for the facebook marketplace classification problem.
    
    Attributes
    ----------
    resnet50 : Model
            Transfer learned resnet50 model downloaded from torch hub.
            
    Methods
    ----------
    forward()
            Override of the forward pass through the resnet50 model.
    '''
    def __init__(self, is_feature_extraction):
        super().__init__()
        # load pre trained resnet50 from torch hub
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # unfreeze the last two layers and retrain entire model   
        for param in self.resnet50.parameters():
            param.requires_grad = False
        if is_feature_extraction:
            self.resnet50.fc = torch.nn.Linear(2048,1000)
            print("Feature extraction turned on")
        else:
            # change final layer of resnet-50 to have an output size = number of possible categories (13 categories)
            self.resnet50.fc = torch.nn.Sequential(torch.nn.Linear(2048,1000),torch.nn.ReLU(),torch.nn.Linear(1000,13))

    def forward(self, X):
        return self.resnet50(X)
