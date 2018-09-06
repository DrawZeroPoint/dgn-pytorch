import numpy as np
import argparse
import torch
import torch.nn as nn
import os
from fat_dataset import load_dataset, utils
from torch.distributions import Normal
import torch.nn.modules.loss as loss

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Generative Network on Falling Things fat_dataset')
    parser.add_argument('--data_dir', type=str, help='location of npy test data', default="./fat_dataset/fat_s-torch")

    args = parser.parse_args()

    test_set = load_dataset.Fat(args.data_dir, 'train', c_st=4, batch_size=20, seq_num=10, b_st=0)

    model_path = os.path.join(".", "model-95000.pt")
    model = torch.load(model_path, map_location=lambda storage, _: storage).to(device)
    if type(model) is nn.DataParallel:
        model = model.module

    test_set.load_new()

    for i in range(10):
        d, c, v = test_set.get_batch()
        d = d.to(device)
        c = c.to(device)
        v = v.to(device)

        y_mu, y_q, r, kld = model(d, c, v)
        subfix = f"0.26train{i}"
        tq = utils.custom_save_img(y_q, f"test_q_{subfix}.png")
        tr = utils.custom_save_img(y_mu, f"test_r_{subfix}.png")

        # elbo = reconstruction + kl_divergence
        # print("elbo: {}, ls: {}".format(elbo, ls.item()))
