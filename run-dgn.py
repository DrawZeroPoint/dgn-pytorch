"""
run-dgn.py

Script to train the a GDN on the FAT fat_dataset.
"""

import os
import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision import transforms

from dgn import DepthGenerativeNetwork
from dataset import custom_save_img
from dataset import FATDataset, RandomCrop, ToTensor

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

commit = '0.21'
data_dir = './dataset/fat_s-torch'
save_dir = './log-{}'.format(commit)
fine_tune = 'none'
batch_size = 8
crop_size = 240

data_parallel = False
gradient_steps = 100000

if __name__ == '__main__':
    print(" - Train id: {}\n"
          " - data_dir: {}\n"
          " - fine_tune model: {}\n"
          " - batch_size: {}\n"
          " - corp_size: {}\n".format(commit, data_dir, fine_tune, batch_size, crop_size))

    fat_dataset = FATDataset("./dataset/fat",
                             "train", trans=transforms.Compose([RandomCrop(crop_size), ToTensor()]))

    dataloader = DataLoader(fat_dataset, batch_size, shuffle=True, num_workers=4)

    # Pixel variance
    sigma_f, sigma_i = 0.7, 2.0

    # Learning rate
    mu_f, mu_i = 5*10**(-5), 5*10**(-4)
    mu, sigma = mu_f, sigma_f

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path = os.path.join(".", fine_tune)
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=lambda storage, _: storage).to(device)
        if type(model) is nn.DataParallel:
            model = model.module
    else:
        # Create model and optimizer
        model = DepthGenerativeNetwork(x_dim=6, y_dim=1, r_dim=256, h_dim=128, z_dim=64,
                                       l_dim=12).to(device)

        # Model optimisations
        model = nn.DataParallel(model) if data_parallel else model

    optimizer = torch.optim.Adam(model.parameters(), lr=mu)

    # Number of gradient steps
    s = 0
    while True:
        if s >= gradient_steps:
            torch.save(model, "model-final.pt")
            break

        for _, batch in enumerate(dataloader):
            img_d = batch['depth'].to(device)
            img_l = batch['left'].to(device)
            img_r = batch['right'].to(device)
            img_cat = torch.cat([img_l, img_r], 1)

            img_d_mu, img_d_q, kld = model(img_d, img_cat)

            # If more than one GPU we must take new shape into account
            batch_size = img_d_q.size(0)

            # Negative log likelihood
            nll = - Normal(img_d_mu, sigma).log_prob(img_d_q)

            reconstruction = torch.mean(nll.view(batch_size, -1), dim=0).sum()
            kl_divergence = torch.mean(kld.view(batch_size, -1), dim=0).sum()

            # Evidence lower bound
            elbo = reconstruction + kl_divergence
            elbo.backward()

            optimizer.step()
            optimizer.zero_grad()

            s += 1

            # Keep a checkpoint every 100 steps
            if s % (gradient_steps/100) == 0:
                torch.save(model, os.path.join(save_dir, "model-{}.pt".format(s)))
                print("model-{}.pt saved.".format(s))

                with torch.no_grad():
                    batch = next(iter(dataloader))
                    img_d = batch['depth'].to(device)
                    img_l = batch['left'].to(device)
                    img_r = batch['right'].to(device)
                    img_cat = torch.cat([img_l, img_r], 1)

                    img_d_mu, img_d_q, kld = model(img_d, img_cat)

                    print("|Steps: {}\t|NLL: {}\t|KL: {}\t|".format(s, reconstruction.item(), kl_divergence.item()))
                    img_show = torch.cat([img_d_q, img_d_mu], 0)
                    custom_save_img(img_show, os.path.join(save_dir, "result_{}.png".format(s)))

                    # Anneal learning rate
                    mu = max(mu_f + (mu_i - mu_f) * (1 - s / (1.6 * 10 ** 6)), mu_f)
                    optimizer.lr = mu * math.sqrt(1 - 0.999 ** s) / (1 - 0.9 ** s)

                    # Anneal pixel variance
                    sigma = max(sigma_f + (sigma_i - sigma_f) * (1 - s / (2 * 10 ** 5)), sigma_f)
