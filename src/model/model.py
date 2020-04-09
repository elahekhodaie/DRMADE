import numpy as np

import torch as t
import torch.nn as nn
from src.model.layers import Encoder, MADE, Decoder
import src.config as config


class DRMADE(nn.Module):
    def __init__(
            self,
            input_size,
            num_channels,
            latent_size,
            made_hidden_layers=config.made_hidden_layers,
            made_natural_ordering=config.made_natural_ordering,
            num_masks=config.made_num_masks,
            num_mix=config.num_mix,
            num_dist_parameters=config.num_dist_parameters,
            distribution=config.distribution,
            parameters_transform=config.parameters_transform,
            parameters_min=config.paramteres_min_value,
            encoder_num_layers=config.encoder_num_layers,
            encoder_layers_activation=config.encoder_layers_activation,
            encoder_latent_activation=config.encoder_latent_activation,
            encoder_latent_bn=config.encoder_bn_latent,
            decoder_num_layers=config.decoder_num_layers,
            decoder_layers_activation=config.decoder_layers_activation,
            decoder_output_activation=config.decoder_output_activation,
            name=None,
    ):
        super(DRMADE, self).__init__()

        assert len(parameters_transform) == num_dist_parameters, 'wrong number of parameter transofrms'
        assert len(parameters_min) == num_dist_parameters, 'wrong number of parameter minimum'

        self.encoder = Encoder(
            num_channels=num_channels,
            latent_size=latent_size,
            input_size=input_size,
            num_layers=encoder_num_layers,
            layers_activation=encoder_layers_activation,
            latent_activation=encoder_latent_activation,
            bn_latent=encoder_latent_bn,
        )
        self.decoder = Decoder(
            num_channels=num_channels,
            latent_size=latent_size,
            output_size=input_size,
            num_layers=decoder_num_layers,
            layers_activation=decoder_layers_activation,
            output_activation=decoder_output_activation,
        )
        self.made = MADE(
            latent_size,
            made_hidden_layers,
            latent_size * (num_dist_parameters if num_mix == 1 else 1 + num_dist_parameters) * num_mix,
            num_masks,
            natural_ordering=made_natural_ordering,
            num_dist_parameters=num_dist_parameters,
            distribution=distribution,
            parameters_transform=parameters_transform,
            parameters_min=parameters_min,
            num_mix=num_mix,
        )

        self.name = 'DRMADE:{}:{}:{}'.format(
            self.encoder.name, self.made.name, self.decoder.name
        ) if not name else name

    def forward(self, x):
        features = self.encoder(x)
        output_image = self.decoder(features)
        output = self.made(features)
        return output, features, output_image

    def num_parameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])

    def save(self, path):
        t.save(self.state_dict(), path)

    def load(self, path, device=None):
        params = t.load(path) if not device else t.load(path, map_location=device)

        added = 0
        for name, param in params.items():
            if name in self.state_dict().keys():
                try:
                    self.state_dict()[name].copy_(param)
                    added += 1
                except Exception as e:
                    print(e)
                    pass
        print('loaded {:.2f}% of params'.format(100 * added / float(len(self.state_dict().keys()))))
