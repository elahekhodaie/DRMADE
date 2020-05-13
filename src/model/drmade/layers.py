import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config as config


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor)


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=config.made_use_biases):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
            self,
            nin,
            hidden_sizes,
            nout,
            num_masks=config.made_num_masks,
            bias=config.made_use_biases,
            natural_ordering=config.made_natural_ordering,
            num_dist_parameters=config.num_dist_parameters,
            distribution=config.distribution,
            parameters_transform=config.parameters_transform,
            parameters_min=config.paramteres_min_value,
            num_mix=config.num_mix,
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

        # seeds for orders/connectivities of the model ensemble
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
        return self.net(x)

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
            parameters.append(
                self.parameters_min[z] + transform(
                    output[:, [j for i in range(self.nin) for j in
                               range(i + self.nin * self.num_mix * z,
                                     self.nin * self.num_mix * (z + 1),
                                     self.nin)]]
                )
            )
        return parameters

    def _log_prob_hitmap(self, x, output=None, parameters=None):
        if output is None:
            output = self(x)
        features = x

        features = features.repeat(1, self.num_mix)[:, self._feature_perm_indexes]
        if parameters is None:
            parameters = self.get_dist_parameters(output)

        dists = self.distribution(*parameters)
        log_probs_dists = dists.log_prob(features).reshape(-1, self.nin, self.num_mix)
        if self.num_mix == 1:
            return log_probs_dists.reshape(-1, self.nin)

        log_mix_coefs = output[:, self._log_mix_coef_perm_indexes].reshape(-1, self.nin, self.num_mix)
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


class Encoder(nn.Module):
    def __init__(
            self,
            num_channels,
            latent_size,
            input_size,
            num_layers,
            bias=config.encoder_use_bias,
            bn_affine=config.encoder_bn_affine,
            bn_eps=config.encoder_bn_eps,
            bn_latent=config.encoder_bn_latent,
            layers_activation=config.encoder_layers_activation,
            latent_activation=config.encoder_latent_activation,
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
        if config.latent_cor_regularization_abs:
            return (torch.abs(correlations)).sum()
        return correlations.sum()

    def latent_distance_regularization(
            self, features, use_norm=config.latent_distance_normalize_features, norm=config.latent_distance_norm
    ):
        batch_size = features.shape[0]
        vec = features
        if use_norm:
            vec = features / ((features ** norm).sum(1, keepdim=True) ** (1 / norm)).repeat(1, self.latent_size)
        a = vec.repeat(1, batch_size).reshape(-1, batch_size, self.latent_size)
        b = vec.repeat(batch_size, 1).reshape(-1, batch_size, self.latent_size)
        return (1 / ((torch.abs(a - b) ** norm + 1).sum(2) ** (1 / norm))).sum()

    def latent_zero_regularization(self, features, eps=config.latent_zero_regularization_eps):
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


class Decoder(nn.Module):
    def __init__(
            self,
            num_channels,
            latent_size,
            output_size,
            num_layers,
            layers_activation=config.decoder_layers_activation,
            output_activation=config.decoder_output_activation,
            bias=config.decoder_use_bias,
            bn_affine=config.decoder_bn_affine,
            bn_eps=config.decoder_bn_eps,
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
            1) + config.decoder_distance_eps) ** (1 / norm)

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
