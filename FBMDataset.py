import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image

class FBMDataset(Dataset):
    def __init__(self, training_data_filepath, img_dir, transform=None ):
        super().__init__()
        print(training_data_filepath)
        self.img_labels = pd.read_csv(training_data_filepath)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_ext = self.img_labels.loc[index, 'id'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_ext)
        label = self.img_labels.iloc[index, 2]  
        image = Image.open(img_path)    
        if self.transform:
            image = self.transform(image)

        return image,label