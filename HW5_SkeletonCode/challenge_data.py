import os
import numpy as np
import pandas as pd
import torch
import skimage.io
import skimage.transform
from torch.utils.data import Dataset, DataLoader
from utils import config, get_device
from PIL import Image
import torchvision.transforms as T

def resize(X):
    image_dim = config('image_dim')
    image_size = (image_dim, image_dim)
    resized = []
    for i in range(X.shape[0]):
        xi = skimage.transform.resize(X[i], image_size, preserve_range=True)
        resized.append(xi)
    return np.array(resized)

class ImageStandardizer(object):
    def __init__(self):
        self.image_mean = None
        self.image_std = None
    def fit(self, X):
        self.image_mean = np.mean(X, axis=(0,1,2))
        self.image_std = np.std(X, axis=(0,1,2))
    def transform(self, X):
        return (X - self.image_mean) / self.image_std

class DogsDataset(Dataset):
    def __init__(self, partition, num_classes=10, transform=None):
        super().__init__()
        if partition not in ['train', 'val', 'test']:
            raise ValueError('Partition {} does not exist'.format(partition))
        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes
        self.transform = transform
        self.metadata = pd.read_csv(config('csv_file'), index_col=0)
        self.X, self.y = self._load_data()
        self.X = resize(self.X)
        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'].dropna().astype(int),
            self.metadata['semantic_label'].dropna()
        ))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        if self.transform:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return image, label
    def _load_data(self):
        if self.partition == 'test':
            if self.num_classes == 5:
                df = self.metadata[self.metadata.partition == self.partition]
            elif self.num_classes == 10:
                df = self.metadata[self.metadata.partition.isin([self.partition, ' '])]
            else:
                raise ValueError('Unsupported test partition: num_classes must be 5 or 10')
        else:
            df = self.metadata[
                (self.metadata.numeric_label < self.num_classes) &
                (self.metadata.partition == self.partition)
            ]
        X, y = [], []
        for i, row in df.iterrows():
            image = skimage.io.imread(os.path.join(config('image_path'), row['filename']))
            label = row['numeric_label']
            X.append(image)
            y.append(label)
        return np.array(X), np.array(y)
    def get_semantic_label(self, numeric_label):
        return self.semantic_labels[numeric_label]

def get_train_val_test_loaders(num_classes):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor()
    ])
    val_test_transform = T.Compose([
        T.ToTensor()
    ])
    tr = DogsDataset('train', num_classes, transform=train_transform)
    va = DogsDataset('val', num_classes, transform=val_test_transform)
    te = DogsDataset('test', num_classes, transform=val_test_transform)
    batch_size = config('cnn.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_dataset(num_classes=10):
    tr = DogsDataset('train', num_classes)
    va = DogsDataset('val', num_classes)
    te = DogsDataset('test', num_classes)
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    te.X = te.X.transpose(0,3,1,2)
    device = get_device()
    tr.X = torch.from_numpy(tr.X).float().to(device)
    va.X = torch.from_numpy(va.X).float().to(device)
    te.X = torch.from_numpy(te.X).float().to(device)
    tr.y = torch.from_numpy(tr.y).long().to(device)
    va.y = torch.from_numpy(va.y).long().to(device)
    te.y = torch.from_numpy(te.y).long().to(device)
    return tr, va, te, standardizer

if __name__ == '__main__':
    np.set_printoptions(precision=2)
    tr, va, te, standardizer = get_train_val_dataset()
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Test:\t", len(te.X))
    print("ImageStandardizer image_mean:", standardizer.image_mean)
    print("ImageStandardizer image_std: ", standardizer.image_std)
