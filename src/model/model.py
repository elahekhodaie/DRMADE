import torch as t
import torch.nn as nn
import torch.nn.functional as F
from src.model.layers import Encoder, MADE
import src.config as config


class DRMADE(nn.Module):
    def __init__(
            self,
            num_channels,
            latent_size,
            num_dist_parameters=config.num_dist_parameters,
            made_hidden_layers=config.made_hidden_layers,
            made_natural_ordering=config.made_natural_ordering,
            num_masks=config.made_num_masks,
    ):
        super(DRMADE, self).__init__()

        self.encoder = Encoder(num_channels, latent_size)
        self.made = MADE(
            latent_size,
            made_hidden_layers,
            latent_size * num_dist_parameters,
            num_masks,
            natural_ordering=made_natural_ordering
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.made(features)
        return output, features
