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


class Oxford_Pets(BaseImageDataset):
    dataset_dir = 'Oxford_Pets'

    def __init__(self, root, verbose=True, **kwargs):
        super(Oxford_Pets, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train, test = self._process_dir(self.dataset_dir)
        if verbose:
            print("=> Oxford_Pets loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test
        self.train_label = []
        for _, label in self.train:
            self.train_label.append(label)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

    def _process_dir(self, root):
        images_train = osp.join(root, 'annotations', 'trainval.txt')
        images_test = osp.join(root, 'annotations', 'test.txt')

        train_data = []
        with open(images_train, 'r', encoding='UTF-8') as f:
            lines_classes = f.readlines()
            for line in lines_classes:
                strs = line.split(' ')
                train_data.append([strs[0], int(strs[1])-1])

        test_data = []
        with open(images_test, 'r', encoding='UTF-8') as f:
            lines_classes = f.readlines()
            for line in lines_classes:
                strs = line.split(' ')
                test_data.append([strs[0], int(strs[1])-1])

        train_dataset = []
        for image_name, label in train_data:
            image_info = (os.path.join(root, 'images', '%s.jpg'% image_name), label)
            train_dataset.append(image_info)

        test_dataset = []
        for image_name, label in test_data:
            image_info = (os.path.join(root, 'images', '%s.jpg'% image_name), label)
            test_dataset.append(image_info)

        return train_dataset, test_dataset


if __name__ == '__main__':
    dataset = Oxford_Pets()
