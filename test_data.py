"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: test_data.py
about: build the validation/test dataset
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from glob import glob
import os


# --- test dataset --- #
class TestData(data.Dataset):
    def __init__(self, test_data_dir, test_data_gt):
        super().__init__()
        fpaths = glob(os.path.join(test_data_dir, '*.jpg'))  # .png

        haze_names = []
        gt_names = []
        for path in fpaths:
            haze_names.append(path.split('/')[-1])
            # gt = path.split('/')[-1].split('_')[0].split('.')[0]
            gt = path.split('\\')[-1].split('.')[0]
            gt_names.append(str(gt)+'.jpg')
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.test_data_dir = test_data_dir
        self.test_data_gt = test_data_gt

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(os.path.join(self.test_data_dir, haze_name.split('\\')[-1])).convert('RGB')  # .convert('RGB')
        gt_img = Image.open(os.path.join(self.test_data_gt, gt_name)).convert('RGB')  # .convert('RGB')

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        haze = transform_haze(haze_img)

        transform_gt = Compose([ToTensor()])
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
