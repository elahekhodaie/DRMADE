import numpy as np

import torch as t
import torch.nn as nn
from src.model.layers import Encoder, MADE
import src.config as config


class DRMADE(nn.Module):
    def __init__(
            self,
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
            latent_tanh=config.encoder_tanh_latent,
            latent_bn=config.encoder_bn_latent,
            name=None,
    ):
        super(DRMADE, self).__init__()

        self.latent_size = latent_size
        self.num_dist_parameters = num_dist_parameters
        self.num_mix = num_mix
        self.distribution = distribution
        self.parameters_transform = parameters_transform
        self.made_hidden_layers = made_hidden_layers
        self.num_masks = num_masks
        self.parameters_min = parameters_min
        self.latent_tanh = latent_tanh
        self.latent_bn = latent_bn

        self.name = 'DRMADE-latent={}{}{}-hl=[{}]-nmasks={}-dist={},nmix={},pmin=[{}]'.format(
            'tanh' if self.latent_tanh else '',
            'bn' if self.latent_bn else '',
            self.latent_size,
            ','.join(str(i) for i in made_hidden_layers),
            self.num_masks,
            self.distribution.__name__,
            self.num_mix,
            ','.join(str(i) for i in self.parameters_min),
        ) if not name else name

        assert len(self.parameters_transform) == num_dist_parameters
        assert len(self.parameters_min) == num_dist_parameters

        self._feature_perm_indexes = [j for i in range(self.latent_size) for j in
                                      range(i, self.latent_size * self.num_mix, self.latent_size)]
        self._log_mix_coef_perm_indexes = [j for i in range(self.latent_size) for j in
                                           range(i + self.latent_size * self.num_mix * self.num_dist_parameters,
                                                 self.latent_size * self.num_mix * (
                                                         self.num_dist_parameters + 1),
                                                 self.latent_size)]
        self.encoder = Encoder(num_channels, latent_size, tanh_latent=self.latent_tanh, bn_latent=latent_bn)
        self.made = MADE(
            latent_size,
            made_hidden_layers,
            latent_size * (num_dist_parameters if num_mix == 1 else 1 + num_dist_parameters) * num_mix,
            num_masks,
            natural_ordering=made_natural_ordering
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.made(features)
        return output, features

    def num_parameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])

    def get_dist_parameters(self, output):
        parameters = []
        for z, transform in enumerate(self.parameters_transform):
            parameters.append(
                self.parameters_min[z] + transform(
                    output[:, [j for i in range(self.latent_size) for j in
                               range(i + self.latent_size * self.num_mix * z,
                                     self.latent_size * self.num_mix * (z + 1),
                                     self.latent_size)]]
                )
            )

        return parameters

    def log_prob_hitmap(self, x, output=None, parameters=None):
        if output is None:
            output = self.made(x)
        features = x

        features = features.repeat(1, self.num_mix)[:, self._feature_perm_indexes]
        if parameters is None:
            parameters = self.get_dist_parameters(output)

        dists = self.distribution(*parameters)
        log_probs_dists = dists.log_prob(features).reshape(-1, self.latent_size, self.num_mix)
        if self.num_mix == 1:
            return log_probs_dists.reshape(-1, self.latent_size)

        log_mix_coefs = output[:, self._log_mix_coef_perm_indexes].reshape(-1, self.latent_size, self.num_mix)
        log_mix_coefs = log_mix_coefs - t.logsumexp(log_mix_coefs, 2, keepdim=True, ).repeat(1, 1, self.num_mix)

        log_probs_dists += log_mix_coefs
        log_probs = t.logsumexp(log_probs_dists, 2)

        return log_probs

    def log_prob(self, *args, **kwargs):
        return t.sum(self.log_prob_hitmap(*args, **kwargs))

    def latent_regularization_term(self, features):
        norm_features = features / ((features ** 2).sum(1, keepdim=True) ** 0.5).repeat(1, self.latent_size)
        return (norm_features @ norm_features.reshape(self.latent_size, -1)).sum()

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
