import torch
import torch.nn as nn
from torchvision import transforms
from skimage import io, transform

import os
from dataset import custom_save_img, FATDataset, RandomCrop, ToTensor
from torch.utils.data import DataLoader


cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda else "cpu")

commit = '0.21'
save_dir = './log-{}'.format(commit)
model = '45000'
batch_size = 1
crop_size = 240


if __name__ == '__main__':
    pre_fix = './dataset/imgs/test'
    name_left = os.path.join(pre_fix, 'left')
    name_right = os.path.join(pre_fix, 'right')
    image_left = io.imread(name_left)
    image_right = io.imread(name_right)
    image_left = transform.resize(image_left, (crop_size, crop_size))
    image_right = transform.resize(image_right, (crop_size, crop_size))

    model_path = os.path.join(save_dir, "model-{}.pt".format(model))
    model = torch.load(model_path, map_location=lambda storage, _: storage).to(device)
    if type(model) is nn.DataParallel:
        model = model.module

    img_l = image_left.to(device)
    img_r = image_right.to(device)
    img_cat = torch.cat([img_l, img_r], 1)

    img_d_mu, img_d_q, kld = model

    img_show = torch.cat([img_d_q, img_d_mu], 0)
    custom_save_img(img_show, os.path.join(save_dir, "test_{}.png".format(i)))
