# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

# !!!The sklearn should be imported before torch
from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dc import datasets
from dc import models
from dc.models.cm import ClusterMemory
from dc.trainers import CCTrainer
from dc.evaluators import Evaluator, extract_features
from dc.utils.data import IterLoader
from dc.utils.data import transforms as T
from dc.utils.data.sampler import RandomMultipleGallerySampler
from dc.utils.data.preprocessor import Preprocessor
from dc.utils.logging import Logger
from dc.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from dc.utils.faiss_rerank import compute_jaccard_distance
from PIL import Image

start_epoch = best_Top1 = 0


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers, num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])   # normalizer比较重要

    # train_transformer = T.Compose([T.RandomResizedCrop(size=height, scale=(0.2, 1.0)),
    #                                # T.Cutout(height, scale=(0, 1.0)),
    #                                # T.HidePatch(height, patch_num=64, hide_prob_scale=(0, 1.0)),
    #                                T.RandomHorizontalFlip(),
    #                                T.GaussianBlur(kernel_size=int(0.1 * height)),
    #                                T.RandomRotation(np.random.uniform(0, 1) * 180),
    #                                T.RandomVerticalFlip(),
    #                                T.ToTensor(),
    #                                normalizer])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = dataset.train if trainset is None else trainset[0]
    train_labels = dataset.train_labels if trainset is None else trainset[2]
    train_filenames = dataset.train_filenames if trainset is None else trainset[1]

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_labels, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer,
                                labels=train_labels, filenames=train_filenames),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader



def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        n, m = dataset.test.shape[0], dataset.train.shape[0]
        random_indices_n, random_indices_m = np.random.permutation(n), np.random.permutation(m)
        testset = np.concatenate((dataset.test[random_indices_n], dataset.train[random_indices_m]), axis=0)
        testset_labels = [dataset.test_labels[i] for i in random_indices_n] + [dataset.train_labels[i] for i in random_indices_m]
        testset_filenames = [dataset.test_filenames[i] for i in random_indices_n] + [dataset.train_filenames[i] for i in random_indices_m]
    else:
        testset_labels, testset_filenames = dataset.train_labels, dataset.train_filenames

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer,
                     labels=testset_labels, filenames=testset_filenames), batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args, checkpoint_exist=False):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    if checkpoint_exist:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model, strip='module.')
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_Top1
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width,
                                  args.batch_size, args.workers)

    # Create model
    model = create_model(args)
    print(model.module.num_features)

    # Evaluator
    evaluator = Evaluator(model)
    print('--------------------------------------------')
    evaluator.evaluate_classification(-1, test_loader, (dataset.test_filenames, dataset.test_labels),
                                      (dataset.train_filenames, dataset.train_labels), K=3)
    print('--------------------------------------------')

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = CCTrainer(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=dataset.train)

            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f in dataset.train_filenames], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
            del cluster_loader

            if epoch == 0:
                print('--------------------------------------------')
                if args.cluster_method == 'dbscan':
                    eps = float(args.eps)
                    print('DBSCAN Clustering criterion: eps: {:.3f}'.format(eps))
                    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
                elif args.cluster_method == 'hdbscan':
                    from hdbscan import HDBSCAN
                    eps = int(args.eps)
                    print('HDBSCAN Clustering criterion: min_cluster_size: {:d}'.format(eps))
                    cluster = HDBSCAN(min_cluster_size=eps, metric='precomputed')
                else:
                    print("There no %s method."%(args.cluster_method))
                print('--------------------------------------------')

            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)

        print('--------------------------------------------')
        evaluator.prediction_for_clustering(epoch, dataset.train_labels, dataset.train_filenames, pseudo_labels, num_cluster)
        evaluator.evaluate_clustering(epoch, dataset.train_labels, pseudo_labels)
        if epoch == 0 or epoch == 10 or epoch == 20 or (epoch == args.epochs - 1):
            X_samples = np.array(features.cpu())
            y_samples = np.array(pseudo_labels)
            n_classes = num_cluster
            evaluator.feature_for_tsne_embedding(epoch, X_samples, y_samples, n_classes, args.logs_dir)
        print('--------------------------------------------')

        del features

        # Create hybrid memory
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset, pseudo_labels_subset, pseudo_labeled_filenames = [], [], []
        for i, (img, fname, label) in enumerate(zip(dataset.train, dataset.train_filenames, pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append(img)
                pseudo_labeled_filenames.append(fname)
                pseudo_labels_subset.append(label.item())

        pseudo_labeled_dataset = np.array(pseudo_labeled_dataset)
        trainset = [pseudo_labeled_dataset, pseudo_labeled_filenames, pseudo_labels_subset]

        train_loader = get_train_loader(
            args, dataset, args.height, args.width, args.batch_size, args.workers, args.num_instances, iters,
            trainset=trainset)
        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        del train_loader

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            print('--------------------------------------------')
            Top1, _, _ = evaluator.evaluate_classification(epoch, test_loader, (dataset.test_filenames, dataset.test_labels),
                                                           (dataset.train_filenames, dataset.train_labels), K=3)
            print('--------------------------------------------')
            is_best = (Top1 > best_Top1)
            # best_Top1 = max(Top1, best_Top1)
            if is_best:
                best_Top1 = Top1
                best_epoch = epoch

            save_or_not = ((epoch + 1) % 1 == 0)
            save_checkpoint({
                'dataset_name': args.dataset,
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_Top1': best_Top1,
            }, is_best, save_or_not, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model Top1: {:5.1%}  best: {:5.1%}  best epoch: {:3d}{}\n'.
                  format(epoch, Top1, best_Top1, best_epoch, ' *' if is_best else ''))


        lr_scheduler.step()

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bridging the Gap Between Supervised and Unsupervised Learning for Fine-grained Image Classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='citrusdisease4',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=256, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # clusterh
    parser.add_argument('--cluster-method', type=str, default='dbscan')
    parser.add_argument('--eps', type=float, default=0.4,
                        help="max neighbor distance for DBSCAN"   # 0.4
                             "it is also used to set the min-cluster-size for HDBSCAN")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet_ibn50a_ori',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)  # 0
    # parser.add_argument('--output', type=int, default=128)  # 0
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")   # 0.2
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--resume', type=str,
                        default="./logs/model_best_0.5.pth.tar",
                        metavar='PATH')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=100)   # 400
    parser.add_argument('--step-size', type=int, default=20)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)   # 10
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='./example',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/citrus_disease_4_resnet_ibn50a'))
    main()
