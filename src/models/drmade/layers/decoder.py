import numpy as np

import torch
import torch.nn as nn

import src.models.drmade.config as model_config
from .utility_layers import Interpolate


class Decoder(nn.Module):
    def __init__(
            self,
            num_channels,
            latent_size,
            output_size,
            num_layers,
            layers_activation=model_config.decoder_layers_activation,
            output_activation=model_config.decoder_output_activation,
            bias=model_config.decoder_use_bias,
            bn_affine=model_config.decoder_bn_affine,
            bn_eps=model_config.decoder_bn_eps,
            name=None
    ):
        super().__init__()
        # Decoder
        num_layers = int(num_layers)
        assert num_layers > 0, 'non-positive number of layers'

        self.latent_image_size = output_size // (2 ** num_layers)
        assert self.latent_image_size > 0, 'number of layers is too large'

        assert output_activation in ['tanh', 'sigmoid'], 'unknown output activation function'
        assert layers_activation in ['relu', 'elu', 'leaky_relu'], 'unknown layers activation function'

        self.output_size = output_size
        self.output_activation = output_activation
        self.layers_activation = layers_activation
        self.num_output_channels = num_channels
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.latent_num_channels = (
                latent_size // (self.latent_image_size * self.latent_image_size) + 1) if latent_size % (
                self.latent_image_size * self.latent_image_size) != 0 else (
                latent_size // (self.latent_image_size * self.latent_image_size))
        self.transform_latent = nn.Linear(latent_size,
                                          self.latent_num_channels * self.latent_image_size * self.latent_image_size,
                                          bias=bias) if latent_size % (
                self.latent_image_size * self.latent_image_size) != 0 else lambda x: x
        self.name = 'Decoder{}{}{}-{}'.format(
            self.num_layers,
            self.layers_activation,
            'bn_affine' if bn_affine else '',
            self.output_activation,
        ) if not name else name

        self.deconv_layers = []
        self.batch_norms = []
        self.layers = []
        self.output_limits = None
        last_size = self.latent_image_size
        for i in range(self.num_layers + 1):
            n_input_channels = 2 ** (5 + self.num_layers - i) if i else self.latent_num_channels
            n_output_channels = 2 ** (4 + self.num_layers - i) if i != self.num_layers else self.num_output_channels
            kernel_size = 6 if (output_size // (2 ** (self.num_layers - i - 1))) == 2 * (last_size + 1) else 5
            if i == self.num_layers:
                kernel_size = 5 + self.output_size - last_size
            last_size = (last_size + 1) * 2 if (output_size // (2 ** (self.num_layers - i - 1))) == 2 * (
                    last_size + 1) else last_size * 2

            self.deconv_layers.append(
                nn.ConvTranspose2d(n_input_channels, n_output_channels, kernel_size, bias=bias, padding=2)
            )

            self.layers.append(self.deconv_layers[i])

            self.batch_norms.append(nn.BatchNorm2d(n_output_channels, eps=bn_eps, affine=bn_affine))
            self.layers.append(self.batch_norms[i])
            if i != self.num_layers:
                if self.output_activation == 'elu':
                    nn.init.xavier_uniform_(self.deconv_layers[i].weight)
                else:
                    nn.init.xavier_uniform_(self.deconv_layers[i].weight,
                                            nn.init.calculate_gain(self.output_activation))
                if self.layers_activation == 'leaky_relu':
                    self.layers.append(nn.LeakyReLU())
                if self.layers_activation == 'elu':
                    nn.init.xavier_uniform_(self.deconv_layers[i].weight)
                    self.layers.append(nn.ELU())
                if self.layers_activation == 'relu':
                    self.layers.append(nn.ReLU())
                self.layers.append(Interpolate(2))
            else:
                nn.init.xavier_uniform_(self.deconv_layers[i].weight, nn.init.calculate_gain(self.output_activation))
                if self.output_activation == 'tanh':
                    self.output_limits = (-1., 1.)
                    self.layers.append(nn.Tanh())
                if self.layers_activation == 'sigmoid':
                    self.output_limits = (0., 1.)
                    self.layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.transform_latent(x)
        x = x.view(x.size(0), self.latent_num_channels, self.latent_image_size, self.latent_image_size)
        x = self.deconv(x)
        return x

    def distance_hitmap(self, input_image, output_image):
        return torch.abs(input_image - output_image)

    def distance(self, input_image, output_image, norm=2):
        return ((self.distance_hitmap(input_image, output_image) ** norm).sum(1).sum(1).sum(
            1) + model_config.decoder_distance_eps) ** (1 / norm)

    def load(self, path, device=None):
        params = torch.load(path) if not device else torch.load(path, map_location=device)

        added = 0
        for name, param in params.items():
            if name in self.state_dict().keys():
                try:
                    self.state_dict()[name].copy_(param)
                    added += 1
                except Exception as e:
                    print(e)
                    pass
        print('loaded {:.2f}% of params decoder'.format(100 * added / float(len(self.state_dict().keys()))))
