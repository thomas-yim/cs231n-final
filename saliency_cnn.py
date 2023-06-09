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




from pathlib import Path
image_path = Path('./Data - Thomas')
def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        '''
        background = Image.new('RGBA', img.size, (255,255,255))
        background.paste(img, mask=img.split()[3])
        return background.convert('RGB')
        '''
        return img.convert('RGBA')

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

transform = T.Compose([
                T.Resize(size=(64, 64)),
                T.ToTensor()
            ])
dataset = dset.ImageFolder(root=image_path, transform=transform, loader=custom_loader)
print(dataset.class_to_idx)
dataset = Subset(dataset, np.arange(25200))
dataset_test = ConcatDataset((Subset(dataset, np.arange(10600, 12600)), Subset(dataset, np.arange(23200,25200))))
test_dataloader = DataLoader(dataset=dataset_test, 
                              batch_size=1)
dataset_test_len = len(dataset_test)
X = []
y = []
# X = torch.empty((0,))
# y = torch.empty((0,))
for batch in test_dataloader:
    inputs, labels = batch
    inputs = inputs.clone().detach().numpy()
    labels = labels.clone().detach().numpy()
    X.append(inputs)
    y.append(labels)
    # X = torch.cat((X, inputs))
    # y = torch.cat((y, labels))
X = torch.tensor(np.array(X)).squeeze()
y = torch.tensor(np.array(y)).squeeze()

print(X.shape)
print(y.shape)
print(y)
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

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

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 4, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    model.eval()
    X.requires_grad_()
    scores = model(X)
    scores = scores.gather(1, y.view(-1, 1)).squeeze()
    scores.backward(torch.ones(scores.size()))
    saliency = X.grad.abs()
    saliency = torch.max(saliency, dim=1)[0]
    print("Saliency done")
    return saliency
saliency = compute_saliency_maps(X, y, model)
saliency = saliency.numpy()
for i in range(len(saliency)):
    image = saliency[i] * 255.0/saliency[i].max()
    print(image)
    saliency_i = Image.fromarray(image)
    saliency_i = saliency_i.convert('L')
    saliency_i.save('Saliency/output_' + str(i) + '.png')