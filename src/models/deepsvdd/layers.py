import torch
from torch.autograd import Function
from torch import nn
import math

import torch
import numpy as np

import torch
from torch.autograd import Variable
from torch import nn


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class AliasMethod:
    '''
        From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''

    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def to(self, device):
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)

    def draw(self, N):
        '''
            Draw N samples from multinomial
        '''
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj


class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, idx, params):
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        # print(x.device, y.device, memory.device, idx.device, params)
        momentum = params[3].item()
        batchSize = x.size(0)
        outputSize = memory.size(0)
        inputSize = memory.size(1)

        # sample positives & negatives
        idx.select(1, 0).copy_(y.to(x.device).data)

        # sample correspoinding weights
        weight = torch.index_select(memory, 0, idx.view(-1))
        weight.resize_(batchSize, K + 1, inputSize)

        # inner product
        out = torch.bmm(weight, x.data.view(batchSize, inputSize, 1))
        out.div_(T).exp_()  # batchSize * self.K+1

        if Z < 0:
            params[2] = out.mean() * outputSize
            Z = params[2].item()
            print("normalization constant Z is set to {:.1f}".format(Z))

        out.div_(Z).resize_(batchSize, K + 1)

        self.save_for_backward(x, memory, y, weight, out, params)
        print('nce func out', out)
        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = gradOutput.size(0)
        # print('gradOut', gradOutput.device ,'weight', weight.device,'x', x.device, 'y', y.device)
        # gradients d Pm / d linear = exp(linear) / Z
        gradOutput.data.mul_(out.data)
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.bmm(gradOutput.data.view(batchSize, 1, K + 1), weight.to(gradOutput.device))
        # print('after bmm')
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = weight.to(gradOutput.device).select(1, 0).resize_as_(x)
        # print('weight pos select')
        weight_pos.mul_(momentum)
        # print('weight pos mul momentum')
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        # print('weight pos add x data 1 - momentum')
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        # print('weight pos norm')
        updated_weight = weight_pos.div(w_norm)
        # print('weight pos / norm', 'updated_weight', updated_weight.device)
        memory.index_copy_(0, y, updated_weight)
        # print('memory index copy')
        print(gradInput.device)
        return gradInput, None, None, None, None


class NCEAverage(nn.Module):
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, Z=None, device='cpu'):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.to(device)
        self.K = K
        self.device = device
        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).to(device).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        batchSize = x.size(0)
        idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1).to(self.device)
        out = NCEFunction.apply(x, y, self.memory, idx.to(x.device), self.params)
        return out


import torch
from torch.autograd import Function
from torch import nn
import math


class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T)  # batchSize * N

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()

        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return gradInput, None, None, None


class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params', torch.tensor([T, momentum]));
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out


class NCECriterion(nn.Module):

    def __init__(self, nLem, eps=1e-7):
        super(NCECriterion, self).__init__()
        self.nLem = nLem
        self.eps = eps

    def forward(self, x, targets):
        batchSize = x.size(0)
        K = x.size(1) - 1
        Pnt = 1 / float(self.nLem)
        Pns = 1 / float(self.nLem)

        # eq 5.1 : P(origin=models) = Pmt / (Pmt + k*Pnt)
        Pmt = x.select(1, 0)
        Pmt_div = Pmt.add(K * Pnt + self.eps)
        lnPmt = torch.div(Pmt, Pmt_div)

        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon_div = x.narrow(1, 1, K).add(K * Pns + self.eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)

        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)

        loss = - (lnPmtsum + lnPonsum) / batchSize

        return loss
