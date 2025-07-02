# Copyright (c) 2020-present.
# All rights reserved.
#
# Date: 2021-07-14
# Author: Jiabao Wang
# Email: jiabao_1108@163.com
#

from __future__ import print_function, absolute_import
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
from collections import OrderedDict
from sklearn.manifold import TSNE
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

from .utils.meters import AverageMeter
from .utils import to_torch


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _) in enumerate(data_loader):
            # print(fnames)
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, test=None, train=None):
    if test is None and train is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f in test], 0)
    y = torch.cat([features[f].unsqueeze(0) for f in train], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.mm(x, y.t())
    # dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #          torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m, x, y


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    num_samples = flat_targets.shape[0]
    num_correct = np.zeros((preds_k, targets_k))

    for c1 in range(preds_k):
        for c2 in range(targets_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


def embedding_tsne(X_sample, y_sample=None):
    embeddings = TSNE(n_components=2, init='pca', random_state=0, verbose=2).fit_transform(X_sample, y_sample)
    return embeddings


def embedding_2dpoints(X_sample, X_embeddings, y_sample, n_classes,
                       embedding_feat2d_points_path):
    num_samples = X_sample.shape[0]
    xx = X_embeddings[:num_samples, 0]
    yy = X_embeddings[:num_samples, 1]
    ll = y_sample[:num_samples]

    CX_embedding = None
    Cy_sample = []
    for i in range(n_classes):
        idx = np.where(y_sample == i)
        idx_feat = X_embeddings[idx, :]
        cent = np.mean(idx_feat, axis=1)
        C_cent = cent
        Cy_sample.append(i)
        if CX_embedding is None:
            CX_embedding = C_cent
        else:
            CX_embedding = np.row_stack((CX_embedding, C_cent))
    cx = CX_embedding[:num_samples, 0]
    cy = CX_embedding[:num_samples, 1]
    cl = Cy_sample

    fig2 = plt.figure(figsize=(12, 9))
    ax = fig2.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, n_classes))
    for i, (x, y, target) in enumerate(zip(xx, yy, ll)):
        if target >= 0 and target < n_classes:
            xc, yc = CX_embedding[target, 0], CX_embedding[target, 1]
            ax.scatter(x, y, color=colors[target], s=3, alpha=0.4, edgecolors=None)
            ax.plot([x, xc], [y, yc], color=colors[target], linewidth=1)
        else:
            ax.scatter(x, y, color='gray', s=3, alpha=0.4, edgecolors=None)

    for i, (x, y, target) in enumerate(zip(cx, cy, cl)):
        if target >= 0 and target < n_classes:
            ax.scatter(x, y, color=colors[target], s=10, marker='*')
            ax.text(x, y + 0.1, '%d' % target, ha='center', va='bottom', fontsize=8)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.tight_layout()

    form = embedding_feat2d_points_path.split('.')[-1]
    plt.savefig(embedding_feat2d_points_path, format=form, dpi=300)


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def feature_extraction_for_evluation(self, data_loader, test, train):
        features, _ = extract_features(self.model, data_loader)
        distmat, test_features, train_features = pairwise_distance(features, test[0], train[0])
        test_ids = np.array([pid for pid in test[1]])
        train_ids = np.array([pid for pid in train[1]])
        return distmat, test_features, train_features, test_ids, train_ids

    def feature_for_tsne_embedding(self, epoch, X_sample, y_sample, n_classes, root_dir):
        X_embeddings = embedding_tsne(X_sample, None)

        embedding_2dpoints(X_sample, X_embeddings, y_sample, n_classes,
                           os.path.join(root_dir, "embedding_feat2d_train_points_%d.pdf" % (epoch)))

    def evaluate_classification(self, epoch, data_loader, test, train, K=5, sigma=0.07):
        distmat, test_features, train_features, test_ids, train_ids = \
            self.feature_extraction_for_evluation(data_loader, test, train)
        C = np.unique(train_ids).shape[0]
        N = test_ids.shape[0]

        dist_topk, idx_topk = distmat.topk(K, dim=1, largest=True, sorted=True)
        train_ids = torch.from_numpy(train_ids)
        train_ids_expand = train_ids.reshape(1, -1).expand(N, -1)
        test_pred_topk = torch.gather(train_ids_expand, 1, idx_topk)

        pred_one_hot = torch.zeros(K, C)
        pred_one_hot.resize_(N * K, C).zero_()
        a = test_pred_topk.view(-1, 1).type(torch.long)
        pred_one_hot.scatter_(1, a, 1)
        dist_tok_t = dist_topk.clone().div_(sigma).exp_()
        probs = torch.sum(torch.mul(pred_one_hot.view(N, -1, C), dist_tok_t.view(N, -1, 1)), 1)
        _, predictions = probs.sort(1, True)

        # Find which predictions match the target
        test_ids = torch.from_numpy(test_ids)
        correct = predictions.eq(test_ids.data.view(-1, 1))

        top1 = correct.narrow(1, 0, 1).sum().item() / N
        topk = correct.narrow(1, 0, K).sum().item() / N

        idx = np.argsort(np.array(distmat), axis=1)[:, -1]
        predictions = train_ids[idx].flatten()

        acc = int((predictions == test_ids).sum()) / float(test_ids.shape[0])
        nmi = metrics.normalized_mutual_info_score(test_ids, predictions)
        ari = metrics.adjusted_rand_score(test_ids, predictions)

        res = {'ACC': acc, 'NMI': nmi, 'ARI': ari, 'Top-1': top1, f"Top-{K}": topk}

        print(f"#Classification Epoch %d, ACC: %.1f%%, NMI: %.1f%%, ARI: %.1f%%, Top-1: %.1f%%, Top-{K}: %.1f%%." % (
            epoch, res['ACC'] * 100, res['NMI'] * 100, res['ARI'] * 100,
            res['Top-1'] * 100, res[f"Top-{K}"] * 100))

        return res['Top-1'], train_features, train_ids

    def evaluate_clustering(self, epoch, labels, predictions):
        targets = np.array(labels)
        # targets = np.array([pid for pid in train])
        num_targets = np.unique(targets).shape[0]

        num_predictions = len(set(predictions)) - (1 if -1 in predictions else 0)
        match = _hungarian_match(predictions, targets, preds_k=num_predictions, targets_k=num_targets)
        reordered_preds = np.zeros(targets.shape[0], dtype=predictions.dtype)
        for pred_i, target_i in match:
            reordered_preds[predictions == int(pred_i)] = int(target_i)

        # Gather performance metrics
        acc = int((reordered_preds == targets).sum()) / float(targets.shape[0])
        nmi = metrics.normalized_mutual_info_score(targets, reordered_preds)
        ari = metrics.adjusted_rand_score(targets, reordered_preds)

        res = {'ACC': acc, 'NMI': nmi, 'ARI': ari}

        print("#Clustering Epoch %d, ACC: %.1f%%, NMI: %.1f%%, ARI: %.1f%%." % (
            epoch, res['ACC'] * 100, res['NMI'] * 100, res['ARI'] * 100))

        return res['ACC']

    def prediction_for_clustering(self, epoch, labels, filenames, pseudo_labels, num_cluster):
        clustered_samples = 0
        cluster_data_idx = [[] for _ in range(num_cluster)]
        print('--------------------------------------------')
        for x in range(num_cluster):
            print(x, end=': {')
            count = 0
            data_idx = []
            for i, (fname, id, y) in enumerate(zip(filenames, labels, pseudo_labels)):
                if (y == x):
                    print(id, end=',')
                    data_idx.append(i)
                    count = count + 1
            print('}', end='#')
            cluster_data_idx[x] = data_idx
            print(count)
            clustered_samples += count

        print('==> Statistics for epoch {}: {} clusters, {} clustered instances, {} un-clustered instances'
              .format(epoch, num_cluster, clustered_samples, pseudo_labels.shape[0] - clustered_samples))
        print('--------------------------------------------')




