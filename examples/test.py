# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

# !!!The sklearn should be imported before torch
from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dc import datasets
from dc import models
from dc.evaluators import Evaluator
from dc.utils.data import transforms as T
from dc.utils.data.preprocessor import Preprocessor
from dc.utils.logging import Logger
from dc.utils.serialization import load_checkpoint, copy_state_dict
from dc.utils.faiss_rerank import compute_jaccard_distance


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.test) | set(dataset.train))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width,
                                  args.batch_size, args.workers)

    # Create model
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model, strip='module.')

    model.cuda()
    model = nn.DataParallel(model)

    # Evaluator
    evaluator = Evaluator(model)
    _, train_features, train_label = evaluator.evaluate_classification(-1, test_loader, dataset.test, dataset.train)
    rerank_dist = compute_jaccard_distance(train_features, k1=args.k1, k2=args.k2)

    # Cluster
    if args.cluster_method == 'dbscan':
        eps = (float)(args.eps)
        print('DBSCAN Clustering criterion: eps: {:.3f}'.format(eps))
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
    elif args.cluster_method == 'hdbscan':
        eps = (int)(args.eps)
        print('HDBSCAN Clustering criterion: min_cluster_size: {:d}'.format(eps))
        from hdbscan import HDBSCAN
        cluster = HDBSCAN(min_cluster_size=eps, metric='precomputed')
    else:
        print("There no %s method." % (args.cluster_method))

    pseudo_labels = cluster.fit_predict(rerank_dist)
    num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    print('==> Statistics: {} clusters'.format(num_cluster))
    # evaluator.prediction_for_clustering(1000, dataset.train, pseudo_labels, num_cluster)
    evaluator.evaluate_clustering(1000, dataset.train, pseudo_labels)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='citrusdisease7')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=256, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet_ibn50a',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    # parser.add_argument('--output', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)

    parser.add_argument('--resume', type=str,
                        default="./logs/model_best.pth.tar",
                        metavar='PATH')

    # cluster
    parser.add_argument('--eps', type=float, default=0.4,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    parser.add_argument('--cluster-method', type=str, default='dbscan')

    # testing configs
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='./example',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    main()
