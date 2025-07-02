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


class Stanford_Cars(BaseImageDataset):
    dataset_dir = 'Stanford_Cars'

    def __init__(self, root, verbose=True, **kwargs):
        super(Stanford_Cars, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train, test = self._process_dir(self.dataset_dir)
        if verbose:
            print("=> Stanford_Cars loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test
        self.train_label = []
        for _, label in self.train:
            self.train_label.append(label)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

    def _process_dir(self, root):
        data_file = osp.join(root, 'cars_annos.mat')

        import scipy.io as sio
        data = sio.loadmat(data_file)
        test_flags = data['annotations']['test'][0]
        labels = data['annotations']['class'][0]
        images = data['annotations']['relative_im_path'][0]

        train_dataset = []
        test_dataset = []
        for test_flag, image_path, label in zip(test_flags, images, labels):
            image_info = (os.path.join(root, str(image_path[0])), int(label[0][0]) - 1)
            if test_flag[0][0]:
                train_dataset.append(image_info)
            else:
                test_dataset.append(image_info)

        return train_dataset, test_dataset


if __name__ == '__main__':
    dataset = Stanford_Cars()
