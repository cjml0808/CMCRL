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


class FGVC_Aircraft(BaseImageDataset):
    dataset_dir = 'FGVC_Aircraft'

    def __init__(self, root, verbose=True, **kwargs):
        super(FGVC_Aircraft, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train, test = self._process_dir(self.dataset_dir)
        if verbose:
            print("=> FGVC_Aircraft loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test
        self.train_label = []
        for _, label in self.train:
            self.train_label.append(label)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

    def _process_dir(self, root):
        classes = osp.join(root, 'data', 'variants.txt')
        images_train = osp.join(root, 'data', 'images_variant_trainval.txt')
        images_test = osp.join(root, 'data', 'images_variant_test.txt')

        class_list = {}
        with open(classes, 'r', encoding='UTF-8') as f:
            i = 0
            lines_classes = f.readlines()
            for line in lines_classes:
                class_list[str(line).replace('\n', '')] = i
                i = i+1

        train_dataset = []
        with open(images_train, 'r', encoding='UTF-8') as f:
            lines_images = f.readlines()
            for line in lines_images:
                image_path = str(line[:7]).strip()+'.jpg'
                label = class_list[str(line[7:-1]).strip()]
                image_info = (os.path.join(root, 'data', 'images', image_path), int(label))
                train_dataset.append(image_info)

        test_dataset = []
        with open(images_test, 'r', encoding='UTF-8') as f:
            lines_train_test = f.readlines()
            for line in lines_train_test:
                image_path = str(line[:7]).strip() + '.jpg'
                label = class_list[str(line[7:-1]).strip()]
                image_info = (os.path.join(root, 'data', 'images', image_path), int(label))
                test_dataset.append(image_info)

        return train_dataset, test_dataset


if __name__ == '__main__':
    dataset = FGVC_Aircraft()


