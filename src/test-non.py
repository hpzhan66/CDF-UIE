import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor
import torch.utils.data as data


class ValDatanon(data.Dataset):
    def __init__(self, dataset_name, val_data_dir):
        super().__init__()
        self.dataset_name = dataset_name
        val_list = os.path.join(val_data_dir, 'data_list.txt')
        with open(val_list) as f:
            contents = f.readlines()
            lowlight_names = [i.strip() for i in contents]

        self.lowlight_names = lowlight_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        lowlight_name = self.lowlight_names[index]
        lowlight_img = Image.open(os.path.join(self.val_data_dir, lowlight_name))

        # Crop the image to ensure dimensions are divisible by 8 (or any other padding)
        padding = 8
        a = lowlight_img.size
        a_0 = a[1] - np.mod(a[1], padding)
        a_1 = a[0] - np.mod(a[0], padding)
        lowlight_crop_img = lowlight_img.crop((0, 0, 0 + a_1, 0 + a_0))

        transform_lowlight = Compose([ToTensor()])
        lowlight_img = transform_lowlight(lowlight_crop_img)

        return lowlight_img, lowlight_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.lowlight_names)


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of CDF-UIE)')
parser.add_argument('-d', '--dataset-name', help='name of dataset',
                    choices=['C60', 'U45', 'UCCS', 'APP'], default='C60')
parser.add_argument('-t', '--test-image-dir', help='test images path', default='./data/classic_test_image/')
parser.add_argument('-c', '--ckpts-dir', help='ckpts path', default='')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-o', '--output-dir', help='output directory to save results', default='./output/appi')
args = parser.parse_args()

val_batch_size = args.val_batch_size
dataset_name = args.dataset_name
output_dir = args.output_dir

# --- Set dataset-specific hyper-parameters  --- #
if dataset_name == 'U45':
    val_data_dir = '/home/non-compare/U45'
elif dataset_name == 'C60':
    val_data_dir = '/home/non-compare/challenging-60'
elif dataset_name == 'APP':
    val_data_dir = '/home/appli'
else:
    val_data_dir = '/home/non-compare/RUIE'


ckpts_dir = args.ckpts_dir

# --- Create output directory if it doesn't exist --- #
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
val_data_loader = DataLoader(ValDatanon(dataset_name, val_data_dir), batch_size=val_batch_size, shuffle=False,
                             num_workers=4)

# --- Define the network --- #
if dataset_name in ['C60', 'U45', 'UCCS', 'APP']:
    from CDF_UIE_arch import InteractNet as CDF_Net
net = CDF_Net()

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(ckpts_dir), strict=False)

# --- Use the evaluation model in testing --- #
net.eval()
print('--- Testing starts! ---')
start_time = time.time()

# --- Process each image and save the output --- #
with torch.no_grad():
    for i, (input_img, filename) in enumerate(val_data_loader):
        input_img = input_img.to(device)
        output_img,out_scale = net(input_img)


        # Save the output image
        save_image(output_img, os.path.join(output_dir, f'{filename[0]}.png'), normalize=True)
        print(f'Processed {filename[0]}')

end_time = time.time() - start_time
print('Testing complete. Time taken: {:.4f} seconds'.format(end_time))