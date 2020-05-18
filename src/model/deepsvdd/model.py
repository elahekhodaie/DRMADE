import torch
import src.config as config
import torch.nn as nn
import torchvision

import numpy as np
from pathlib import Path
from src.model.deepsvdd.layers import NCEAverage, LinearAverage, NCECriterion


class DeepSVDD(nn.Module):
    def __init__(
            self,
            train_data,
            latent_size,
            norm=2,
            norm_eps=1e-10,
            nce_t=0.07,
            nce_k=0,
            nce_m=0,
            device='cpu',
    ):
        super(DeepSVDD, self).__init__()
        self.train_data = train_data
        self.latent_size = latent_size
        self.norm = norm
        self.norm_eps = norm_eps

        self.nce_t = nce_t
        self.nce_k = nce_k
        self.nce_m = nce_m

        # initializing network
        self.resnet = torchvision.models.resnet18().to(device)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, latent_size).to(device)

        # initializing center & memory
        if self.nce_k > 0:
            self.lemniscate = NCEAverage(latent_size, len(train_data), nce_k, nce_t, nce_m, device)
        else:
            self.lemniscate = LinearAverage(latent_size, len(train_data), nce_t, nce_m)

        if hasattr(self.lemniscate, 'K'):
            self.criterion = NCECriterion(len(train_data)).to(device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(device)

        self.center = None

        self.name = 'Resnet18-{}:t{}k{}m{}'.format(
            self.latent_size,
            self.nce_t,
            self.nce_k,
            self.nce_m,
        )

    def forward(self, input):
        return self.resnet(input)

    def init_center(self, device, init_zero=False):
        with torch.no_grad():
            self.center = torch.zeros(self.latent_size).to(device)
            if init_zero:
                return
            temp_loader = self.train_data.get_dataloader(100, shuffle=False)
            for images, output in temp_loader:
                images = images.to(device)
                self.center += self(images).sum(axis=0) / images.shape[0]
            self.center = self.center / len(temp_loader)

    def init_memory(self, device):
        with torch.no_grad():
            self.memory = torch.zeros(len(self.train_data), self.latent_size).to(device)
            self.memory.uniform_()
            self.memory = self.memory / ((self.memory ** self.norm).sum(axis=1, keepdim=True) ** (1 / self.norm))
        self.memory.requires_grad = False

    def log_prob_hitmap(self, images, indexes, features=None):
        if features is None:
            features = self(images)
        features = self._normalize_features(features)
        old_features = self.memory[indexes]
        cosine_similarity = (features * old_features) / self.temperature
        normalizing_constant = torch.exp(
            (features @ self.memory.transpose(0, 1)) / self.temperature).sum(axis=1, keepdim=True)
        return cosine_similarity - torch.log(normalizing_constant)

    def log_prob(self, images, indexes, features=None):
        return self.log_prob_hitmap(images, indexes, features).sum()

    def _normalize_features(self, features):
        features = features - self.center
        return features / (features ** self.norm).sum(axis=1, keepdim=True) ** (1 / self.norm)

    def iteration_difference_hitmap(self, images, indexes, features=None):
        if features is None:
            features = self(images)
        features = self._normalize_features(features)
        old_features = self.memory[indexes]
        return (((features - old_features) ** self.norm).sum(axis=1, keepdim=True) + self.norm_eps) ** (1 / self.norm)

    def iteration_difference(self, images, indexes, features=None):
        return self.iteration_difference_hitmap(images, indexes, features).sum()

    def update_memory(self, images, indexes, features=None):
        with torch.no_grad():
            if features is None:
                features = self(images)
            features = self._normalize_features(features)
            self.memory[indexes] = features.clone().detach()
            self.memory.requires_grad = False

    def radius_hitmap(self, images, features=None, ):
        if features is None:
            features = self(images)
        return (((features - self.center) ** self.norm).sum(axis=1, keepdim=True) + self.norm_eps) ** (1 / self.norm)

    def radius(self, images, features=None, ):
        return self.radius_hitmap(images, features, ).sum()

    def classify(self, images, top=5):
        with torch.no_grad():
            features = self._normalize_features(self(images))
            cosine_similarities = torch.exp((features @ self.memory.transpose(0, 1)) / self.temperature)
            values, indexes = cosine_similarities.topk(self.k)
            indexes = indexes.cpu()
            result = list()
            for i in range(images.shape[0]):
                values, counts = np.unique(self.train_data.labels[indexes[i]], return_counts=True)
                arg_sort = np.argsort(counts)[::-1]
                result.append(values[arg_sort][:top])
        return result

    def save(self, tag):
        torch.save(self.state_dict(), f'{tag}.pth')

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
        print('loaded {:.2f}% of params deepsvdd'.format(100 * added / float(len(self.state_dict().keys()))))
