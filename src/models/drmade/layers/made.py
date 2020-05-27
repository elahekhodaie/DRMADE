import numpy as np

import torch
import torch.nn as nn

import src.models.drmade.config as model_config
from .utility_layers import MaskedLinear


class MADE(nn.Module):
    def __init__(
            self,
            nin,
            hidden_sizes,
            nout,
            num_masks=model_config.made_num_masks,
            bias=model_config.made_use_biases,
            natural_ordering=model_config.made_natural_ordering,
            num_dist_parameters=model_config.num_dist_parameters,
            distribution=model_config.distribution,
            parameters_transform=model_config.parameters_transform,
            parameters_min=model_config.paramteres_min_value,
            num_mix=model_config.num_mix,
            name=None,
    ):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to drmade ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don'torch use random permutations
        """

        super().__init__()
        assert nout % nin == 0, "nout must be integer multiple of nin"

        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.distribution = distribution
        self.parameters_min = parameters_min
        self.num_dist_parameters = num_dist_parameters
        self.parameters_transform = parameters_transform
        self.num_mix = num_mix
        self.num_masks = num_masks
        self._feature_perm_indexes = [j for i in range(self.nin) for j in
                                      range(i, self.nin * self.num_mix, self.nin)]
        self._log_mix_coef_perm_indexes = [j for i in range(self.nin) for j in
                                           range(i + self.nin * self.num_mix * self.num_dist_parameters,
                                                 self.nin * self.num_mix * (
                                                         self.num_dist_parameters + 1),
                                                 self.nin)]

        self.name = 'MADEhl=[{}]-nmasks={}-dist={},nmix={},pmin=[{}]'.format(
            ','.join(str(i) for i in hidden_sizes),
            self.num_masks,
            self.distribution.__name__,
            self.num_mix,
            ','.join(str(i) for i in self.parameters_min),
        ) if not name else name

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1, bias),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the models ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        output = self.net(x)
        if output.requires_grad:
            output.register_hook(
                lambda grad: torch.where(torch.isnan(grad) + torch.isinf(grad), torch.zeros_like(grad), grad))
        return output

    def test_autoregressive_property(self):
        rng = np.random.RandomState(14)
        x = (rng.rand(1, self.nin) > 0.5).astype(np.float32)
        # run backpropagation for each dimension to compute what other
        # dimensions it depends on.
        res = []
        for k in range(self.nout):
            xtr = torch.from_numpy(x)
            xtr.requires_grad = True
            xtrhat = self(xtr)
            loss = xtrhat[0, k]
            loss.backward()

            depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            isok = k % self.nin not in depends_ix

            res.append((len(depends_ix), k, depends_ix, isok))

        # pretty print the dependencies
        res.sort()
        for nl, k, ix, isok in res:
            print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))

    def get_dist_parameters(self, output):
        parameters = []
        for z, transform in enumerate(self.parameters_transform):
            parameters.append(self.parameters_min[z] + transform(
                output[:, [j for i in range(self.nin) for j in
                           range(i + self.nin * self.num_mix * z, self.nin * self.num_mix * (z + 1), self.nin)]]))

        return parameters

    def _log_prob_hitmap(self, x, output=None, parameters=None):
        if output is None:
            output = self(x)
        features = x

        features = features.repeat(1, self.num_mix)[:, self._feature_perm_indexes]
        if features.requires_grad:
            features.register_hook(
                lambda grad: torch.where(torch.isnan(grad) + torch.isinf(grad), torch.zeros_like(grad), grad))
        if parameters is None:
            parameters = self.get_dist_parameters(output)

        dists = self.distribution(*parameters)

        log_probs_dists = dists.log_prob(features).view(-1, self.nin, self.num_mix)

        if self.num_mix == 1:
            return log_probs_dists.view(-1, self.nin)

        log_mix_coefs = output[:, self._log_mix_coef_perm_indexes].view(-1, self.nin, self.num_mix)
        log_mix_coefs = log_mix_coefs - torch.logsumexp(log_mix_coefs, 2, keepdim=True, ).repeat(1, 1, self.num_mix)

        log_probs_dists += log_mix_coefs
        log_probs = torch.logsumexp(log_probs_dists, 2)

        return log_probs

    def log_prob_hitmap(self, features, outputs=None, parameters=None):
        result = self._log_prob_hitmap(features, outputs, parameters)
        if outputs is not None or parameters is not None:
            return result
        self.update_masks()
        for i in range(self.num_masks - 1):
            result += self._log_prob_hitmap(features).clone()
            self.update_masks()
        return result / self.num_masks

    def log_prob(self, features, output=None, parameters=None):
        result = 0.
        result += self._log_prob_hitmap(features, output, parameters).sum()
        if output is not None or parameters is not None:
            return result
        self.update_masks()
        for i in range(self.num_masks - 1):
            result += self._log_prob_hitmap(features).sum()
            self.update_masks()
        return result / self.num_masks

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
        print('loaded {:.2f}% of params made'.format(100 * added / float(len(self.state_dict().keys()))))
