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
            parameters_transform=config.parameters_transform
    ):
        super(DRMADE, self).__init__()

        self.latent_size = latent_size
        self.num_dist_parameters = num_dist_parameters
        self.num_mix = num_mix
        self.distribution = distribution
        self.parameters_transform = parameters_transform

        assert len(self.parameters_transform) == num_dist_parameters
        self._feature_perm_indexes = [j for i in range(self.latent_size) for j in
                                      range(i, self.latent_size * self.num_mix, self.latent_size)]
        self._log_mix_coef_perm_indexes = [j for i in range(self.latent_size) for j in
                                           range(i + self.latent_size * self.num_mix * self.num_dist_parameters,
                                                 self.latent_size * self.num_mix * (
                                                         self.num_dist_parameters + 1),
                                                 self.latent_size)]
        self.encoder = Encoder(num_channels, latent_size)
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

    def log_prob_hitmap(self, x, output=None):
        if output is None:
            output, features = self(x)
        else:
            features = x
        features = features.repeat(1, self.num_mix)[:, self._feature_perm_indexes]

        parameters = []
        for z, transform in enumerate(self.parameters_transform):
            parameters.append(transform(output[:, [j for i in range(self.latent_size) for j in
                                                   range(i + self.latent_size * self.num_mix * z,
                                                         self.latent_size * self.num_mix * (z + 1),
                                                         self.latent_size)]]))

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
