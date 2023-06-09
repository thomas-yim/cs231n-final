import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import functools

import torchvision.datasets as dset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

def get_transform(grayscale=False, normalize=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    
    transform_list.append(transforms.Resize(256, transforms.InterpolationMode.BICUBIC))


    transform_list += [transforms.ToTensor()]
    if normalize:
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)



class FakesDataset(Dataset):
    def __init__(self, eevee_path, depth_path, normal_path, cycles_path, fake_path):
        files = os.listdir(eevee_path)
        self.eevee_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.fake_paths = []
        self.cycles_paths = []

        self.transform = get_transform(grayscale=False)
        self.transform_gray = get_transform(grayscale=True)
        self.transform_true = get_transform(grayscale=False, normalize=False)

        excluded_mugs = ["002", "020", "050", "078", "088", "099", "110", "115", "117"]
        for file in sorted(files):
            if file.split("_")[1] not in excluded_mugs:
                self.eevee_paths.append(eevee_path + "/" + file)
                self.depth_paths.append(depth_path + "/" + file)
                self.normal_paths.append(normal_path + "/" + file)
                self.cycles_paths.append(cycles_path + "/" + file)
        
        fake_files = os.listdir(fake_path)
        for file in sorted(fake_files):
            if file.split("_")[-2] == "fake":
                self.fake_paths.append(fake_path + "/" + file)
        
        print(len(self.eevee_paths))
        print(len(self.depth_paths))
        print(self.depth_paths)
        print(len(self.normal_paths))
        print(len(self.fake_paths))
        print(len(self.cycles_paths))
        
    def __len__(self):
        return len(self.eevee_paths)

    def __getitem__(self, idx):
        eevee_path = self.eevee_paths[idx]
        depth_path = self.depth_paths[idx]
        normal_path = self.normal_paths[idx]
        fake_path = self.fake_paths[idx]
        cycles_path = self.cycles_paths[idx]
        
        eevee_img = self.transform(Image.open(eevee_path).convert("RGB"))
        depth_img = self.transform_gray(Image.open(depth_path).convert("L"))
        normal_img = self.transform(Image.open(normal_path).convert("RGB"))
        cycles_img = self.transform_true(Image.open(cycles_path).convert("RGB"))
        fake_img = self.transform(Image.open(fake_path).convert("RGB"))
        


        input_stacked = torch.cat((eevee_img, depth_img, normal_img, fake_img), dim=0)

        return input_stacked, cycles_img


dataset = FakesDataset("data/stacked_pix2pix/EEVEE/test", "data/stacked_pix2pix/DEPTH/test", "data/stacked_pix2pix/NORMAL/test", "data/stacked_pix2pix/CYCLES/test", "data/stacked_pix2pix/FAKES/")
dataloader = DataLoader(dataset=dataset, batch_size=1)

X = []
real_images = []
y = []

print(len(dataset))

for i, batch in enumerate(dataloader):
    input, real = batch
    label = 1
    input = input.clone().detach().numpy()
    X.append(input[0])
    real_images.append(real[0])
    y.append(label)

X = torch.tensor(np.array(X)).squeeze()
y = torch.tensor(np.array(y)).squeeze()

print(X.shape)
print(y.shape)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

state_dict = torch.load("latest_net_D.pth")
new_dict = {}
for key, value in state_dict.items():
    newKey = key[6:]
    new_dict[newKey] = value
model = NLayerDiscriminator(10, 64, n_layers=3, norm_layer=nn.BatchNorm2d).model

model.load_state_dict(new_dict)

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    
criterionGAN = GANLoss("vanilla")

def compute_saliency_map(x, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    model.eval()
    x.requires_grad_()
    
    score = model(x)
    loss = criterionGAN(score, False)

    loss.backward(torch.tensor(1))
    saliency = x.grad.abs()
    saliency = torch.max(saliency[0][-3:], dim=0, keepdim=True)[0]
    print("Saliency done")
    return saliency

# cur_img = X[0][-3:]
# print(X[0][-3:].size())
# print(cur_img.numpy().shape)
# transform = T.ToPILImage()
# real_image = transform(cur_img)
# real_image.show()
# print(real_image)
# real_image = real_image.convert('RGB')
# real_image.save('Saliency_old/real_' + str(0) + '.png')

transform = transforms.ToPILImage()
for j in range(len(X)):
    saliency = compute_saliency_map(X[j:j+1], y, model)
    saliency = saliency.numpy()

    # pil_image = transform(real_images[j])
    # pil_image.save("Saliency_discriminator/real/real_image_" + str(j) + ".png")

    for i in range(len(saliency)):
        image = (saliency[i]-saliency[i].min()) * 255.0/(saliency[i].max()-saliency[i].min())
        saliency_i = Image.fromarray(image)
        saliency_i = saliency_i.convert('L')
        saliency_i.save('Saliency_discriminator/normal/output_' + str(j) + '.png')