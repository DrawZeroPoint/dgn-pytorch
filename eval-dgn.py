import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import os
from dataset import custom_save_img, FATDataset, RandomCrop, ToTensor,  Rescale
from torch.utils.data import DataLoader


cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda else "cpu")

commit = '0.5'
save_dir = './log-{}'.format(commit)
model = '50000'
batch_size = 1
crop_size = 240


if __name__ == '__main__':
    test_set = FATDataset("./dataset/fat", "test",
                          trans=transforms.Compose([RandomCrop(crop_size), Rescale(crop_size), ToTensor()]))

    dataloader = DataLoader(test_set, batch_size, shuffle=True, num_workers=1)

    model_path = os.path.join(save_dir, "model-{}.pt".format(model))
    model = torch.load(model_path).to(device)  # , map_location=lambda storage, _: storage
    if type(model) is nn.DataParallel:
        model = model.module

    for i, batch in enumerate(dataloader):
        img_d = batch['depth'].to(device)
        img_l = batch['left'].to(device)
        img_r = batch['right'].to(device)
        img_cat = torch.cat([img_l, img_r], 1)

        img_d_hat = model.sample(img_cat)

        img_show = torch.cat([img_d, img_d_hat], 0)
        custom_save_img(img_show, os.path.join(save_dir, "test_{}.png".format(i)), batch_size)
        if i == 10:
            break
