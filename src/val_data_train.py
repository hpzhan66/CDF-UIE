# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Validation/test dataset --- #
class ValData_train(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'data_list.txt'#'final_test_datalist.txt'eval15/datalist.txt
        with open(val_list) as f:
            contents = f.readlines() 
            degrad_names = [i.strip() for i in contents]
            gt_names = degrad_names
 
        self.degrad_names = degrad_names
        self.gt_names = gt_names 
        self.val_data_dir = val_data_dir
        self.data_list=val_list

    @staticmethod
    def pad_to_divisible(img, downscale_factor):
        _, h, w = img.size()
        # Calculate padding
        pad_h = (downscale_factor - h % downscale_factor) % downscale_factor
        pad_w = (downscale_factor - w % downscale_factor) % downscale_factor
        # Apply padding
        img_padded = F.pad(img, (0, pad_w, 0, pad_h), "constant", 0)  # Padding order: (left, right, top, bottom)
        return img_padded
    def get_images(self, index):
        degrad_name = self.degrad_names[index]
        gt_name = self.gt_names[index]
        degrad_img = Image.open(self.val_data_dir + 'input/' + degrad_name)
        gt_img = Image.open(self.val_data_dir + 'gt/' + gt_name)
        transform_degrad = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])     
        degrad = transform_degrad(degrad_img)
        gt = transform_gt(gt_img)
        downscale_factor = 2  # Adjust as necessary
        degrad_img_padded = self.pad_to_divisible(degrad, downscale_factor)
        gt_img_padded = self.pad_to_divisible(gt, downscale_factor)
        return degrad_img_padded, gt_img_padded,degrad_name #

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self): 
        return len(self.degrad_names)
 