import os
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class StockImagesDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]), self.img_labels.iloc[idx, 1])
        image = read_image(img_path)[:1].float()
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
if __name__ == '__main__':
    dataset = StockImagesDataset(
        annotations_file='data/train_annotations_20.csv',
        img_dir='images/20'
    )


    image, label = dataset[0]
    plt.title(f"Label: {label}")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image.squeeze(0), cmap='gray')
    plt.savefig("test.png")


    
