import numpy as np

import torch
import torch.nn as nn

import src.models.drmade.config as model_config


class Encoder(nn.Module):
    def __init__(
            self,
            num_channels,
            latent_size,
            input_size,
            num_layers,
            bias=model_config.encoder_use_bias,
            bn_affine=model_config.encoder_bn_affine,
            bn_eps=model_config.encoder_bn_eps,
            bn_latent=model_config.encoder_bn_latent,
            layers_activation=model_config.encoder_layers_activation,
            latent_activation=model_config.encoder_latent_activation,
            name=None,
    ):
        super(Encoder, self).__init__()
        assert num_layers > 0, 'non-positive number of layers'
        latent_image_size = input_size // (2 ** num_layers)

        assert latent_image_size > 0, 'number of layers is too large'
        assert latent_activation in ['', 'tanh', 'leaky_relu', 'sigmoid'], 'unknown latent activation'
        assert layers_activation in ['relu', 'elu', 'leaky_relu'], 'non-positive number of layers'

        self.num_layers = num_layers
        self.num_input_channels = num_channels
        self.latent_size = latent_size
        self.bn_latent = bn_latent
        self.latent_activation = latent_activation
        self.layers_activation = layers_activation

        self.name = 'Encoder{}{}{}-{}{}{}'.format(
            self.num_layers,
            self.layers_activation,
            'bn_affine' if bn_affine else '',
            self.latent_size,
            self.latent_activation,
            'bn' if self.bn_latent else '',
        ) if not name else name

        self.conv_layers = []
        self.batch_norms = []
        self.layers = []
        for i in range(self.num_layers):
            self.conv_layers.append(
                nn.Conv2d(32 * (2 ** (i - 1)) if i else self.num_input_channels, 32 * (2 ** i), 5, bias=bias,
                          padding=2))
            if self.layers_activation == 'elu':
                nn.init.xavier_uniform_(self.conv_layers[i].weight)
            else:
                nn.init.xavier_uniform_(self.conv_layers[i].weight, nn.init.calculate_gain(self.layers_activation))
            self.layers.append(self.conv_layers[i])

            self.batch_norms.append(nn.BatchNorm2d(32 * (2 ** i), eps=bn_eps, affine=bn_affine))
            self.layers.append(self.batch_norms[i])
            if self.layers_activation == 'leaky_relu':
                self.layers.append(nn.LeakyReLU())
            if self.layers_activation == 'elu':
                self.layers.append(nn.ELU())
            if self.layers_activation == 'relu':
                self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2, 2))

        self.convs = nn.Sequential(*self.layers)

        self.fc1 = nn.Linear(32 * (2 ** (num_layers - 1)) * (latent_image_size ** 2), self.latent_size, bias=bias)
        if not self.latent_activation and self.latent_activation:
            nn.init.xavier_uniform_(self.fc1.weight, nn.init.calculate_gain(self.latent_activation))

        self.output_limits = None
        self.activate_latent = (lambda x: x)
        if self.latent_activation == 'tanh':
            self.output_limits = (-1, 1.)
            self.activate_latent = (lambda x: torch.tanh(x))
        elif self.activate_latent == 'leaky_relu':
            self.activate_latent = (lambda x: torch.nn.functional.leaky_relu(x))
        elif self.activate_latent == 'sigmoid':
            self.activate_latent = (lambda x: torch.nn.functional.sigmoid(x))
            self.output_limits = (0.,)
        self.latent_bn = nn.BatchNorm1d(self.latent_size, eps=bn_eps, affine=bn_affine) if self.bn_latent else lambda \
                x: x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.latent_bn(x)
        x = self.activate_latent(x)
        return x

    def latent_cor_regularization(self, features):
        norm_features = features / ((features ** 2).sum(1, keepdim=True) ** 0.5).repeat(1, self.latent_size)
        correlations = norm_features @ norm_features.reshape(self.latent_size, -1)
        if model_config.latent_cor_regularization_abs:
            return (torch.abs(correlations)).sum()
        return correlations.sum()

    def latent_distance_regularization(
            self, features, use_norm=model_config.latent_distance_normalize_features,
            norm=model_config.latent_distance_norm
    ):
        batch_size = features.shape[0]
        vec = features
        if use_norm:
            vec = features / ((features ** norm).sum(1, keepdim=True) ** (1 / norm)).repeat(1, self.latent_size)
        a = vec.repeat(1, batch_size).reshape(-1, batch_size, self.latent_size)
        b = vec.repeat(batch_size, 1).reshape(-1, batch_size, self.latent_size)
        return (1 / ((torch.abs(a - b) ** norm + 1).sum(2) ** (1 / norm))).sum()

    def latent_zero_regularization(self, features, eps=model_config.latent_zero_regularization_eps):
        return torch.sum(1.0 / (eps + torch.abs(features)))

    def latent_var_regularization(self, features):
        return torch.sum(((features - features.sum(1, keepdim=True) / self.latent_size) ** 2).sum(1) / self.latent_size)

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
        print('loaded {:.2f}% of params encoder'.format(100 * added / float(len(self.state_dict().keys()))))
