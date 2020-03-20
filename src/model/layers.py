import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config as config


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
    def __init__(self, nin, hidden_sizes, nout, num_masks=config.made_num_masks, bias=config.made_use_biases,
                 natural_ordering=config.made_natural_ordering):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

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


class Encoder(nn.Module):
    def __init__(
            self,
            num_channels,
            latent_size,
            bias=config.encoder_use_bias,
            bn_affine=config.encoder_bn_affine,
            bn_eps=config.encoder_bn_eps,
            tanh_latent=config.encoder_tanh_latent,
            bn_latent=config.encoder_bn_latent,
    ):
        super(Encoder, self).__init__()

        self.num_input_channels = num_channels
        self.latent_size = latent_size
        self.tanh_latent = tanh_latent
        self.bn_latent = bn_latent

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(self.num_input_channels, 32, 5, bias=bias, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=bn_eps, affine=bn_affine)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=bias, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=bn_eps, affine=bn_affine)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=bias, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=bn_eps, affine=bn_affine)
        self.fc1 = nn.Linear(128 * 3 * 3, self.latent_size, bias=bias)
        if tanh_latent:
            nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
        if bn_latent:
            self.bn1d = nn.BatchNorm1d(self.latent_size, eps=bn_eps, affine=bn_affine)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.tanh_latent:
            x = torch.tanh(x)
        if self.bn_latent:
            x = self.bn1d(x)
        return x
