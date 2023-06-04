import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import random


class StackedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with stacked channel features (depth and normals)


    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot + "/EEVEE", opt.phase)  # create a path '/path/to/data/trainA'
        self.dir_A_depth = os.path.join(opt.dataroot + '/DEPTH', opt.phase)
        self.dir_A_normal = os.path.join(opt.dataroot + '/NORMAL', opt.phase)
        self.dir_B = os.path.join(opt.dataroot + '/CYCLES', opt.phase)  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_normal_paths = sorted(make_dataset(self.dir_A_normal, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_depth_paths = sorted(make_dataset(self.dir_A_depth, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        assert len(self.A_paths) == len(self.A_normal_paths) and len(self.A_paths) == len(self.A_depth_paths)
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform = get_transform(self.opt)
        self.transform_depth = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_depth_path = self.A_depth_paths[index % self.A_size]  # make sure index is within then range
        A_normal_path = self.A_normal_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]

        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)
        
        A_img = self.transform(Image.open(A_path)) # already RBG
        A_depth_img = self.transform_depth(Image.open(A_depth_path).convert("L"))
        A_normal_img = self.transform(Image.open(A_normal_path).convert("RGB"))

        A_stacked = torch.cat((A_img, A_depth_img, A_normal_img), dim=0)
        B_img = self.transform(Image.open(B_path))

        return {'A': A_stacked, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
