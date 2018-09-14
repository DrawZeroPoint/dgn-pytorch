import torch
import torch.nn as nn
from skimage import io

import os
from dataset import custom_save_img
import cv2
import numpy as np


cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda else "cpu")

commit = '0.5'
save_dir = './log-{}'.format(commit)
model = '77000'
batch_size = 1
crop_size = 240


if __name__ == '__main__':
    pre_fix = './dataset'
    name_left = os.path.join(pre_fix, 'left2.png')
    name_right = os.path.join(pre_fix, 'right2.png')
    image_left = io.imread(name_left)
    image_right = io.imread(name_right)
    img_l = cv2.resize(image_left, (crop_size, crop_size))
    img_r = cv2.resize(image_right, (crop_size, crop_size))

    model_path = os.path.join(save_dir, "model-{}.pt".format(model))
    model = torch.load(model_path, map_location=lambda storage, _: storage).to(device)
    if type(model) is nn.DataParallel:
        model = model.module

    img_l = img_l.astype(np.float32).transpose((2, 0, 1))
    img_r = img_r.astype(np.float32).transpose((2, 0, 1))
    img_l = torch.from_numpy(img_l).mul(0.0039215686).to(device)
    img_r = torch.from_numpy(img_r).mul(0.0039215686).to(device)
    img_cat = torch.cat([img_l, img_r], 0).unsqueeze(0)

    with torch.no_grad():
        img_d_mu = model.sample(img_cat)

        custom_save_img(img_d_mu, os.path.join(save_dir, "test.png"), 1)

        img_d = img_d_mu.cpu().squeeze(0).numpy().transpose((1, 2, 0))
        img_d = cv2.resize(img_d, (image_left.shape[1], image_left.shape[0]))  # w, h
        cv2.imwrite(os.path.join(save_dir, "test_cv.png"), img_d*255)
