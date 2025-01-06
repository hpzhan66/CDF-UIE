# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import torch
import os

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, dataset_name,val_data_dir):
        super().__init__() 
        self.dataset_name = dataset_name
        val_list = os.path.join(val_data_dir, 'data_list.txt')
        with open(val_list) as f:
            contents = f.readlines()
            degrad_names = [i.strip() for i in contents]
            if self.dataset_name=='UFO' or self.dataset_name=='UIEB'or self.dataset_name=='LSUI':
                gt_names = degrad_names #
            else:
                gt_names = None 
                print('The dataset is not included in this work.')  
        self.degrad_names = degrad_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.data_list=val_list
    def get_images(self, index):
        degrad_name = self.degrad_names[index]
        padding = 8
        # build the folder of validation/test data in our way
        if os.path.exists(os.path.join(self.val_data_dir, 'input')):
            degrad_img = Image.open(os.path.join(self.val_data_dir, 'input', degrad_name))
            if os.path.exists(os.path.join(self.val_data_dir, 'gt')) :
                gt_name = self.gt_names[index]
                gt_img = Image.open(os.path.join(self.val_data_dir, 'gt', gt_name)) ##   
                a = degrad_img.size

                a_0 =a[1] - np.mod(a[1],padding)
                a_1 =a[0] - np.mod(a[0],padding)            
                degrad_crop_img = degrad_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = gt_img.crop((0, 0, 0 + a_1, 0+a_0))
                transform_degrad = Compose([ToTensor()])
                transform_gt = Compose([ToTensor()])
                degrad_img = transform_degrad(degrad_crop_img)
                gt_img = transform_gt(gt_crop_img)
            else: 
                # the inputs are used to calculate PSNR.
                a = degrad_img.size
                a_0 =a[1] - np.mod(a[1],padding)
                a_1 =a[0] - np.mod(a[0],padding)            
                degrad_crop_img = degrad_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = degrad_crop_img
                transform_degrad = Compose([ToTensor() ])
                transform_gt = Compose([ToTensor()])
                degrad_img = transform_degrad(degrad_crop_img)
                gt_img = transform_gt(gt_crop_img) 
        # Any folder containing validation/test images
        else:
            degrad_img = Image.open(os.path.join(self.val_data_dir, degrad_name))
            a = degrad_img.size
            a_0 =a[1] - np.mod(a[1],padding)
            a_1 =a[0] - np.mod(a[0],padding)            
            degrad_crop_img = degrad_img.crop((0, 0, 0 + a_1, 0+a_0))
            gt_crop_img = degrad_crop_img
            transform_degrad = Compose([ToTensor()])
            transform_gt = Compose([ToTensor()])
            degrad_img = transform_degrad(degrad_crop_img)
            gt_img = transform_gt(gt_crop_img)           
        return degrad_img, gt_img, degrad_name


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.degrad_names)


# --- Test dataset --- #
class TestData(data.Dataset):
    def __init__(self, dataset_name, test_data_dir):
        super().__init__()
        self.dataset_name = dataset_name
        test_list = os.path.join(test_data_dir, 'data_list.txt')
        with open(test_list) as f:
            contents = f.readlines()
            degrad_names = [i.strip() for i in contents]
            if self.dataset_name in ['UHD', 'LOLv1', 'LOLv2', 'UIEB']:
                gt_names = degrad_names
            else:
                gt_names = None
                print('The dataset is not included in this work.')
        self.degrad_names = degrad_names
        self.gt_names = gt_names
        self.test_data_dir = test_data_dir
        self.data_list = test_list

    def get_images(self, index):
        degrad_name = self.degrad_names[index]
        resize_dim = (256, 256)

        # Build the folder of test data in our way
        if os.path.exists(os.path.join(self.test_data_dir, 'input')):
            degrad_img = Image.open(os.path.join(self.test_data_dir, 'input', degrad_name))
            if os.path.exists(os.path.join(self.test_data_dir, 'gt')):
                gt_name = self.gt_names[index]
                gt_img = Image.open(os.path.join(self.test_data_dir, 'gt', gt_name))

                degrad_img = degrad_img.resize(resize_dim)
                gt_img = gt_img.resize(resize_dim)

                transform = Compose([ToTensor()])
                degrad_img = transform(degrad_img)
                gt_img = transform(gt_img)
            else:
                degrad_img = degrad_img.resize(resize_dim)
                gt_img = degrad_img

                transform = Compose([ToTensor()])
                degrad_img = transform(degrad_img)
                gt_img = transform(gt_img)
        else:
            degrad_img = Image.open(os.path.join(self.test_data_dir, degrad_name))
            degrad_img = degrad_img.resize(resize_dim)
            gt_img = degrad_img

            transform = Compose([ToTensor()])
            degrad_img = transform(degrad_img)
            gt_img = transform(gt_img)

        return degrad_img, gt_img, degrad_name
    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.degrad_names)