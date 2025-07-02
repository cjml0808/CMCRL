# encoding: utf-8
# Copyright (c) 2020-present.
# All rights reserved.
#
# Date: 2021-07-14
# Author: Jiabao Wang
# Email: jiabao_1108@163.com
#

import os
import os.path as osp
from ..utils.data import BaseImageDataset


class Stanford_Dogs(BaseImageDataset):
    dataset_dir = 'Stanford_Dogs'

    def __init__(self, root, verbose=True, **kwargs):
        super(Stanford_Dogs, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train, test = self._process_dir(self.dataset_dir)
        if verbose:
            print("=> Stanford_Dogs loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test
        self.train_label = []
        for _, label in self.train:
            self.train_label.append(label)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

    def _process_dir(self, root):
        train_file = osp.join(root, 'lists', 'train_list.mat')
        test_file = osp.join(root, 'lists', 'test_list.mat')

        import scipy.io as sio
        train_data = sio.loadmat(train_file)
        test_data = sio.loadmat(test_file)

        train_dataset = []
        for image_path, label in zip(train_data['file_list'].tolist(), train_data['labels'].tolist()):
            image_info = (os.path.join(root, 'Images', str(image_path[0][0])), int(label[0])-1)
            train_dataset.append(image_info)

        test_dataset = []
        for image_path, label in zip(test_data['file_list'].tolist(), test_data['labels'].tolist()):
            image_info = (os.path.join(root, 'Images', str(image_path[0][0])), int(label[0])-1)
            test_dataset.append(image_info)

        return train_dataset, test_dataset


if __name__ == '__main__':
    dataset = Stanford_Dogs()
    for idx, x in enumerate(dataset.train):
        if idx % 500 == 0:
            print(x)
    print("--------------------------------")
    for idx, x in enumerate(dataset.test):
        if idx % 500 == 0:
            print(x)
