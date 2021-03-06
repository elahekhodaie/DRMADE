import numpy as np
from pathlib import Path

import torch as t
import torch.nn as nn
from src.models.drmade.layers import Encoder, MADE, Decoder
import src.models.drmade.config as model_config

class DRMADE(nn.Module):
    def __init__(
            self,
            input_size,
            num_channels,
            latent_size,
            made_hidden_layers=model_config.made_hidden_layers,
            made_natural_ordering=model_config.made_natural_ordering,
            num_masks=model_config.made_num_masks,
            num_mix=model_config.num_mix,
            num_dist_parameters=model_config.num_dist_parameters,
            distribution=model_config.distribution,
            parameters_transform=model_config.parameters_transform,
            parameters_min=model_config.paramteres_min_value,
            encoder_num_layers=model_config.encoder_num_layers,
            encoder_layers_activation=model_config.encoder_layers_activation,
            encoder_latent_activation=model_config.encoder_latent_activation,
            encoder_latent_bn=model_config.encoder_bn_latent,
            decoder_num_layers=model_config.decoder_num_layers,
            decoder_layers_activation=model_config.decoder_layers_activation,
            decoder_output_activation=model_config.decoder_output_activation,
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
        Path(path).mkdir(parents=True, exist_ok=True)
        t.save(self.state_dict(), f'{path}/drmade.pth')
        t.save(self.encoder.state_dict(), f'{path}/encoder.pth')
        t.save(self.decoder.state_dict(), f'{path}/decoder.pth')
        t.save(self.made.state_dict(), f'{path}/made.pth')

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
        print('loaded {:.2f}% of params drmade'.format(100 * added / float(len(self.state_dict().keys()))))

    def forward_ae(self, x, features=None):
        if not features:
            features = self.encoder(x)
        return self.decoder(features)

    def forward_drmade(self, x, features=None):
        if not features:
            features = self.encoder(x)
        return self.made(features)
