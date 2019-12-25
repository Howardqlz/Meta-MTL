from torch.utils.data import Dataset
from torchvision.datasets.cifar import CIFAR100
import os, sys, pickle
import numpy as np


class MyCIFAR(Dataset):

    def __init__(self, dataset, extra_labels=[]):
        self.dataset = dataset
        self.extra_labels = extra_labels

    def __getitem__(self, index):
        img, label = self.dataset[index]
        labels = [label]+[label_type[index] for label_type in self.extra_labels]
        return img, labels

    def __len__(self):
        return len(self.dataset)


class MyCIFAR100(CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, fine_labels=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                elif fine_labels:
                    self.targets.extend(entry['fine_labels'])
                else:
                    self.targets.extend(entry['coarse_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if not fine_labels:
            self.meta['key'] = "coarse_label_names"
        self._load_meta()