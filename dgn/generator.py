"""
The inference-generator architecture is conceptually
similar to the encoder-decoder pair seen in variational
autoencoders. The difference here is that the model
must infer latents from a cascade of time-dependent inputs
using convolutional and recurrent networks.

Additionally, a representation vector is shared between
the networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

SCALE = 4  # Scale of image generation process


class Conv2dLSTMCell(nn.Module):
    """
    2d convolutional long short-term memory (LSTM) cell.
    Functionally equivalent to nn.LSTMCell with the
    difference being that nn.Kinear layers are replaced
    by nn.Conv2D layers.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of image kernel
    :param stride: length of kernel stride
    :param padding: number of pixels to pad with
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, data_input, states):
        """
        Send input through the cell.

        :param data_input: input to send through
        :param states: (hidden, cell) pair of internal state
        :return new (hidden, cell) pair
        """
        (hidden, cell) = states

        forget_gate = torch.sigmoid(self.forget(data_input))
        input_gate = torch.sigmoid(self.input(data_input))
        output_gate = torch.sigmoid(self.output(data_input))
        state_gate = torch.tanh(self.state(data_input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell


class GeneratorNetwork(nn.Module):
    """
    Network similar to a convolutional variational
    auto encoder that refines the generated image
    over a number of iterations.

    :param y_dim: number of channels in input depth image
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param l_dim: Number of refinements of density
    """

    def __init__(self, y_dim, r_dim, z_dim=64, h_dim=128, l_dim=12):
        super(GeneratorNetwork, self).__init__()
        self.L = l_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Core computational units
        self.inference_core = Conv2dLSTMCell(h_dim + y_dim + r_dim, h_dim, kernel_size=5, stride=1, padding=2)
        self.generator_core = Conv2dLSTMCell(r_dim + z_dim, h_dim, kernel_size=5, stride=1, padding=2)

        # Inference, prior
        self.posterior_density = nn.Conv2d(h_dim, 2 * z_dim, kernel_size=5, stride=1, padding=2)
        self.prior_density = nn.Conv2d(h_dim, 2 * z_dim, kernel_size=5, stride=1, padding=2)

        # Generative density
        self.observation_density = nn.Conv2d(h_dim, y_dim, kernel_size=1, stride=1, padding=0)

        # Up/down-sampling primitives
        self.upsample = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0)
        self.downsample = nn.Conv2d(y_dim, y_dim, kernel_size=SCALE, stride=SCALE, padding=0)

    def forward(self, d, r):
        """
        Attempt to reconstruct d with corresponding context representation r.

        :param d: depth image to send through
        :param r: representation for image
        :return mean of distribution that generate d and kl-divergence
        """
        batch_size, _, h, w = d.size()
        kl = 0

        # Increase dimensions
        if r.size(2) != h // SCALE:
            r = r.repeat(1, 1, h // SCALE, w // SCALE)

        # Reset hidden state
        hidden_g = d.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        hidden_i = d.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))

        # Reset cell state
        cell_g = d.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        cell_i = d.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))

        u = d.new_zeros((batch_size, self.h_dim, h, w))

        d = self.downsample(d)

        for _ in range(self.L):
            # Prior factor (eta π network)
            o = self.prior_density(hidden_g)
            p_mu, p_std = torch.split(o, self.z_dim, dim=1)
            # softplus f(x)=ln(1+e^x)
            prior_distribution = Normal(p_mu, F.softplus(p_std))

            # Inference state update
            hidden_i, cell_i = self.inference_core(torch.cat([r, hidden_g, d], dim=1), [hidden_i, cell_i])

            # Posterior factor (eta e network)
            o = self.posterior_density(hidden_i)
            q_mu, q_std = torch.split(o, self.z_dim, dim=1)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))

            # Posterior sample
            z = posterior_distribution.rsample()

            # Calculate u
            hidden_g, cell_g = self.generator_core(torch.cat([r, z], dim=1), [hidden_g, cell_g])
            u = self.upsample(hidden_g) + u

            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)

        y_mu = self.observation_density(u)

        return y_mu, kl

    def sample(self, r):
        """
        Sample from the prior distribution to generate
        a new image given a viewpoint and representation
        :param r: representation (context)
        """
        batch_size, h, w = r.size()[0], r.size()[2], r.size()[3]

        device = torch.device("cuda:1")
        hidden_g = torch.zeros((batch_size, self.h_dim, h, w)).to(device)
        cell_g = torch.zeros((batch_size, self.h_dim, h, w)).to(device)
        u = torch.zeros((batch_size, self.h_dim, h * SCALE, w * SCALE)).to(device)

        for _ in range(self.L):
            o = self.prior_density(hidden_g)
            p_mu, p_log_std = torch.split(o, self.z_dim, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_log_std))

            # Prior sample
            z = prior_distribution.sample()

            # Calculate u rz or zr ?
            hidden_g, cell_g = self.generator_core(torch.cat([z, r], dim=1), [hidden_g, cell_g])
            u = self.upsample(hidden_g) + u

        y_mu = self.observation_density(u)

        return y_mu
