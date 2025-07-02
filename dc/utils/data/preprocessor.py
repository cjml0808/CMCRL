from __future__ import absolute_import

import os.path as osp
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, labels=None, filenames=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.labels = labels
        self.filenames = filenames

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid = self.filenames[index], self.labels[index]
        img = self.dataset[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, index


# class Preprocessor(Dataset):
#     def __init__(self, dataset, score=None, root=None, transform=None, train=True):
#         super(Preprocessor, self).__init__()
#         self.dataset = dataset
#         self.root = root
#         self.transform = transform
#         self.train = train
#         if self.train:
#             self.score = score
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, indices):
#         return self._get_single_item(indices)
#
#     def _get_single_item(self, index):
#         fname, pid = self.dataset[index]
#         fpath = fname
#         if self.root is not None:
#             fpath = osp.join(self.root, fname)
#         img = Image.open(fpath).convert('RGB')
#
#         if self.train:
#             score_name, _ = self.score[index]
#             score = np.load(score_name, encoding='bytes', allow_pickle=True)
#
#             if self.transform is not None:
#                 img, score = self.transform([img, score])
#         else:
#             if self.transform is not None:
#                 img = self.transform(img)
#
#         return img, fname, pid, index