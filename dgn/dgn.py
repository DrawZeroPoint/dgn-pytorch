import random

import torch
import torch.nn as nn
from torch.distributions import Normal

from .representation import TowerRepresentation
from .generator import GeneratorNetwork


class DepthGenerativeNetwork(nn.Module):
    """
    :param x_dim: number of channels in input rgb image
    :param y_dim: number of channels in input depth image
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param l_dim: Number of refinements of density
    """

    def __init__(self, x_dim, y_dim, r_dim, h_dim, z_dim, l_dim=12):
        super(DepthGenerativeNetwork, self).__init__()
        self.r_dim = r_dim

        self.representation = TowerRepresentation(x_dim, r_dim)
        self.generator = GeneratorNetwork(y_dim, r_dim, z_dim, h_dim, l_dim)

    def forward(self, depth, rgb_cat):
        """
        Forward through the DGN.
        :param depth: batch of depth images [b, 1, h, w]
        :param rgb_cat: batch of color images [b, 6, h, w]
        """
        # representation generated from input images
        # B 256 120 120
        phi = self.representation(rgb_cat)

        y_q = depth
        y_mu, kl = self.generator(y_q, phi)

        # Return reconstruction and query viewpoint
        # for computing error
        return [y_mu, y_q, kl]

    def sample(self, rgb_cat, sigma=2.0):
        """
        Sample from the network given some context and viewpoint.
        :param rgb_cat: batch of color images [b, 6, h, w]
        :param sigma: pixel variance
        """
        phi = self.representation(rgb_cat)
        y_mu = self.generator.sample(phi)
        return y_mu
