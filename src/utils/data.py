from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import src.config as config
import numpy as np

default_transform = transforms.Compose([transforms.ToTensor(), config.input_rescaling])


class DatasetSelection(Dataset):
    def __init__(self,
                 dataset,
                 root=config.data_dir,
                 train=True,
                 classes=None,
                 transform=default_transform,
                 target_transform=lambda x: x,
                 download=True,
                 return_indexes=False
                 ):
        self.transform=transform
        self.target_transform=target_transform
        self.return_indexes = return_indexes
        self.whole_data = dataset(root, train, transform=transform, target_transform=target_transform,
                                  download=download)
        self.data = self.whole_data
        if classes is not None:
            self.data = [(data, label) for data, label in self.whole_data if label in classes]
            self.labels = np.array([label for data, label in self.whole_data if label in classes])
        else:
            self.labels = np.array([label for data, label in self.whole_data])

        if return_indexes:
            self.data = [(data, (label, index)) for index, (data, label) in enumerate(self.data)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def input_shape(self):
        return self.data[0][0].shape

    def get_dataloader(self, batch_size,
                       shuffle=config.dataloader_shuffle,
                       num_workers=config.dataloader_num_workers,
                       pin_memory=config.dataloader_pin_memory,
                       drop_last=config.dataloader_drop_last,
                       sampler=None
                       ):
        return DataLoader(self, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory,
                          drop_last=drop_last, sampler=sampler)
