import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision.io import read_image

class FBMDataset(Dataset):
    def __init__(self, training_data_filepath, img_dir, transform=None ):
        super().__init__()
        self.img_labels = pd.read_csv(training_data_filepath)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.loc[index, 'id'] + '.jpg')
        image = read_image(img_path)
        label = self.img_labels.iloc[index, 2]      
        if self.transform:
            image = self.transform(image)

        return image.float(),label
    
dataset = FBMDataset("training_data.csv","../images_fb/images/cleaned_images")
print(dataset.__getitem__(0)[0].shape)
print(dataset.__getitem__(0)[1])