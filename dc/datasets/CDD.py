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
from ..utils.data.base_dataset import BaseImageDataset
import pickle
from PIL import Image
import numpy as np


class CDD(BaseImageDataset):
    dataset_dir = 'cdd'

    def __init__(self, root, verbose=True, **kwargs):
        super(CDD, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train, test = self._process_dir(self.dataset_dir)
        if verbose:
            print("=> CDD loaded")
            self.num_train_pids, self.num_train_imgs, self.num_test_pids, self.num_test_imgs = \
                self.print_dataset_statistics(train, test)

        self.train = train["data"]
        self.test = test["data"]
        self.train_labels = list(train["labels"])
        self.test_labels = list(test["labels"])
        self.train_filenames = list(train["filenames"])
        self.test_filenames = list(test["filenames"])
        self.class_num = train["class_num"]
        self.label_names = train["label_names"]


    def _process_dir(self, root):
        # now load the picked numpy arrays
        file_path = os.path.join(root, "train")
        with open(file_path, "rb") as f:
            train_dataset = pickle.load(f, encoding="latin1")

        file_path = os.path.join(root, "test")
        with open(file_path, "rb") as f:
            test_dataset = pickle.load(f, encoding="latin1")

        return train_dataset, test_dataset

# class Citrus_disease_7(BaseImageDataset):
#     dataset_dir = 'citrus_disease_7'
#
#     def __init__(self, root, verbose=True, **kwargs):
#         super(Citrus_disease_7, self).__init__()
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         train, test = self._process_dir(self.dataset_dir)
#         if verbose:
#             print("=> Citrus_disease_7 loaded")
#             self.print_dataset_statistics(train, test)
#
#         self.train = train
#         self.test = test
#         self.train_label = []
#         for _, label in self.train:
#             self.train_label.append(label)
#
#         self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
#         self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)
#
#     def _process_dir(self, root):
#         images = osp.join(root, 'images.txt')
#         image_class_label = osp.join(root, 'image_class_labels.txt')
#         train_test_split = osp.join(root, 'train_test_split(0.8).txt')
#
#         image_list = []
#         with open(images, 'r', encoding='UTF-8') as f:
#             lines_images = f.readlines()
#             for line in lines_images:
#                 strs = line.split(' ')
#                 image_list.append(str(strs[1]).replace('\n',''))
#
#         class_list = []
#         with open(image_class_label, 'r', encoding='UTF-8') as f:
#             lines_classes = f.readlines()
#             for line in lines_classes:
#                 strs = line.split(' ')
#                 class_list.append(int(strs[1]))
#         # print(set(class_list))
#
#         train_test_list = []
#         with open(train_test_split, 'r', encoding='UTF-8') as f:
#             lines_train_test = f.readlines()
#             for line in lines_train_test:
#                 strs = line.split(' ')
#                 train_test_list.append(int(strs[1]))
#
#         train_dataset = []
#         test_dataset = []
#         for image_path, label, train_test in zip(image_list, class_list, train_test_list):
#             if train_test > 0:
#                 image_info = (os.path.join(root, 'images', image_path), int(label))
#                 train_dataset.append(image_info)
#             else:
#                 image_info = (os.path.join(root, 'images', image_path), int(label))
#                 test_dataset.append(image_info)
#
#         return train_dataset, test_dataset



if __name__ == '__main__':
    dataset = CDD()
    for idx, x in enumerate(dataset.train):
        if idx % 500 == 0:
            print(x)
    print("--------------------------------")
    for idx, x in enumerate(dataset.test):
        if idx % 500 == 0:
            print(x)