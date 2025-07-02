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


class Oxford_Flowers(BaseImageDataset):
    dataset_dir = 'Oxford_Flowers'

    def __init__(self, root, verbose=True, **kwargs):
        super(Oxford_Flowers, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train, test = self._process_dir(self.dataset_dir)
        if verbose:
            print("=> Oxford_Flowers loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test
        self.train_label = []
        for _, label in self.train:
            self.train_label.append(label)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

    def _process_dir(self, root):
        data_labels = osp.join(root, 'imagelabels.mat')
        data_split = osp.join(root, 'setid.mat')

        import scipy.io as sio
        mat_labels = sio.loadmat(data_labels)
        labels_list = mat_labels['labels'][0].tolist()
        mat_split = sio.loadmat(data_split)
        train_data = mat_split['trnid'][0].tolist()
        train_data.extend(mat_split['valid'][0].tolist())
        test_data = mat_split['tstid'][0].tolist()

        train_dataset = []
        for data_id in train_data:
            image_info = (os.path.join(root, 'jpg', 'image_%05d.jpg'% data_id), labels_list[data_id-1]-1)
            train_dataset.append(image_info)

        test_dataset = []
        for data_id in test_data:
            image_info = (os.path.join(root, 'jpg', 'image_%05d.jpg'% data_id), labels_list[data_id-1]-1)
            test_dataset.append(image_info)

        return train_dataset, test_dataset


if __name__ == '__main__':
    dataset = Oxford_Flowers()
