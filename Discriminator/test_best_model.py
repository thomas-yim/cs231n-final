import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
import imageio
from collections import Counter
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


from pathlib import Path
def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        '''
        background = Image.new('RGBA', img.size, (255,255,255))
        background.paste(img, mask=img.split()[3])
        return background.convert('RGB')
        '''
        return img.convert('RGBA')

transform = T.Compose([
                T.Resize(size=(64, 64)),
                T.ToTensor()
            ])
dataset = dset.ImageFolder(root="/images", transform=transform, loader=custom_loader)
print(dataset.class_to_idx)
test_dataloader = DataLoader(dataset=dataset, 
                              batch_size=100)
dataset_test_len = len(dataset)
def check_accuracy_part34(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))



channel_1 = 64
channel_2 = 32
channel_3 = 16
conv1 = nn.Sequential(nn.Conv2d(4, channel_1, kernel_size=5, padding=2, bias=True),
    nn.BatchNorm2d(channel_1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(0.1))
conv2 = nn.Sequential(nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1, bias=True),
    nn.BatchNorm2d(channel_2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(0.1))
conv3 = nn.Sequential(nn.Conv2d(channel_2, channel_3, kernel_size=3, padding=1, bias=True),
    nn.BatchNorm2d(channel_3),
    nn.ReLU(),
    nn.MaxPool2d(4),
    nn.Dropout(0.1))
model = nn.Sequential(
    conv1,
    conv2,
    conv3,
    Flatten(),
    nn.Linear(channel_3 * 4 * 4, 2)
)
model.load_state_dict(torch.load('./best_model.pth'))
check_accuracy_part34(test_dataloader, model)