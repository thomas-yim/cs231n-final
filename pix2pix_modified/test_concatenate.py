import torch
from PIL import Image
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms

A_path = "data/eevee/mug_000_000.png"
A_depth_path = "data/depth/mug_000_000.png"
A_normal_path = "data/normal/mug_000_000.png"

transform_list = []

transform_list.append(transforms.ToTensor())
transform_custom =  transforms.Compose(transform_list)

A_img = transform_custom(Image.open(A_path)) # already RGB
A_depth_img = transform_custom(Image.open(A_depth_path).convert("L"))
A_normal_img = transform_custom(Image.open(A_normal_path).convert('RGB'))
print(A_img.size())
print(A_depth_img.size())
print(A_normal_img.size())
print(torch.mean(A_normal_img, dim=(1,2)))
A_stacked = torch.cat((A_img, A_normal_img), dim=0)
print(A_stacked.size())