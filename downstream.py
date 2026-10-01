import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
from torchvision import models
import logging
import pickle
import time
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import random

from citrus_data import CitrusDisease6
from citrus_data_7 import CitrusDisease7, CitrusDisease4
from dc import models

import warnings
warnings.filterwarnings("ignore")


resize_size_dict = {
    'imagenet': 256,
    'tiny-imagenet': 74,
    'cifar10': 40,
    'cifar100': 40
}
crop_size_dict = {
    'imagenet': 224,
    'tiny-imagenet': 64,
    'cifar10': 32,
    'cifar100': 32
}
seed = 2

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def eval():
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    num_class = 7
    model = models.create('resnet_ibn50a_4wa', num_features=2048, norm=True, dropout=0,
                          num_classes=num_class, pooling_type='gem')
    checkpoint_exist = True
    PATH = f"./CMCRL/examples/logs/citrus_disease_7_resnet_ibn50a/"
    if checkpoint_exist:
        checkpoint = torch.load(os.path.join(PATH, f"model_10_4head_512(4wa).pth.tar"), map_location=device)
        state_dict = checkpoint['state_dict']
        model = copy_state_dict(state_dict, model, strip='module.')

    logging.basicConfig(filename=os.path.join(PATH, f"validation_{checkpoint['epoch']}.log"),
                        level=logging.DEBUG)
    logging.info(f"Using device: {device}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Loading model from the folder {PATH.split('/')[-1]}.")

    if num_class == 7:
        k = 5
        train_loader, val_loader = get_citrusdisease7_data_loaders()
    print(f"Dataset: citrusdisease{num_class}")
    logging.info(f"Dataset: citrusdisease{num_class}")

    for name, param in model.named_parameters():
        if name.split('.')[0] == 'base':
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    epochs = 100
    best_top1 = 0
    best_topk = 0
    best_top1_epoch = 0
    best_topk_epoch = 0
    best_f1, best_r, best_p = 0, 0, 0
    best_f1_epoch, best_r_epoch, best_p_epoch = 0, 0, 0
    logging.info(f"Validation with dataset: citrusdisease{num_class}.")
    begin = time.time()
    for epoch in range(epochs):
        epoch_begin = time.time()
        top1_train_accuracy = 0
        conf_matrix = np.zeros((num_class, num_class), dtype=np.int64)
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            if x_batch.shape[0] > 1:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
            elif x_batch.shape[0] == 1:
                x_batch = torch.cat([x_batch, x_batch], dim=0).to(device)
                y_batch = torch.cat([y_batch, y_batch], dim=0).to(device)

            model.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]
            batch_cm = confusion_matrix(y_batch.cpu().numpy(), logits.argmax(dim=1).cpu().numpy(),
                                        labels=list(range(num_class)))
            conf_matrix += batch_cm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)

        TP = np.diag(conf_matrix)
        FP = conf_matrix.sum(axis=0) - TP
        FN = conf_matrix.sum(axis=1) - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        precision, recall, f1 = precision_recall_f1_from_counts(TP, FP, FN)
        f1_train = f1.mean()
        r_train = recall.mean()
        p_train = precision.mean()

        model.eval()
        top1_accuracy = 0
        topk_accuracy = 0
        conf_matrix_val = np.zeros((num_class, num_class), dtype=np.int64)
        with torch.no_grad():
            for counter, (x_batch, y_batch) in enumerate(val_loader):
                if x_batch.shape[0] > 1:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                elif x_batch.shape[0] == 1:
                    x_batch = torch.cat([x_batch, x_batch], dim=0).to(device)
                    y_batch = torch.cat([y_batch, y_batch], dim=0).to(device)

                d = time.time()

                logits = model(x_batch)

                top1, topk = accuracy(logits, y_batch, topk=(1, k))
                top1_accuracy += top1[0]
                topk_accuracy += topk[0]
                batch_cm_val = confusion_matrix(y_batch.cpu().numpy(), logits.argmax(dim=1).cpu().numpy(),
                                                 labels=list(range(num_class)))
                conf_matrix_val += batch_cm_val

        top1_accuracy /= (counter + 1)
        topk_accuracy /= (counter + 1)
        TP_val = np.diag(conf_matrix_val)
        FP_val = conf_matrix_val.sum(axis=0) - TP_val
        FN_val = conf_matrix_val.sum(axis=1) - TP_val
        TN_val = conf_matrix_val.sum() - (TP_val + FP_val + FN_val)
        precision_val, recall_val, _f1_val = precision_recall_f1_from_counts(TP_val, FP_val, FN_val)
        f1_val = _f1_val.mean()
        r_val = recall_val.mean()
        p_val = precision_val.mean()

        model.train()

        if epoch == 0 or top1_accuracy > best_top1:
            best_top1 = top1_accuracy
            best_top1_epoch = epoch
            best_top1_model = model
        if epoch == 0 or topk_accuracy > best_topk:
            best_topk = topk_accuracy
            best_topk_epoch = epoch
        if epoch == 0 or f1_val > best_f1:
            best_f1 = f1_val
            best_f1_epoch = epoch
            best_f1_model = model
        if epoch == 0 or r_val > best_r:
            best_r = r_val
            best_r_epoch = epoch
        if epoch == 0 or p_val > best_p:
            best_p = p_val
            best_p_epoch = epoch
        epoch_time = time.time() - epoch_begin

        print(
            f"Epoch {epoch}\tLoss: {loss}\tTop1 Train accuracy {top1_train_accuracy.item()}"
            f"\tF1 train score {f1_train}\tRecall train score {r_train}\tPrecision train score {p_train}"
            f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTopk val acc: {topk_accuracy.item()}"
            f"\tF1 val score: {f1_val}\tRecall val score: {r_val}\tPrecision val score: {p_val}"
            f"\tEpoch time:{epoch_time}")
        logging.info(
            f"Epoch {epoch}\tLoss: {loss}\tTop1 Train accuracy {top1_train_accuracy.item()}"
            f"\tF1 train score {f1_train}\tRecall train score {r_train}\tPrecision train score {p_train}"
            f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTopk val acc: {topk_accuracy.item()}"
            f"\tF1 val score: {f1_val}\tRecall val score: {r_val}\tPrecision val score: {p_val}"
            f"\tEpoch time:{epoch_time}")
    all_time = time.time() - begin
    print(f"Best_top1_accuracy: {best_top1}.")
    print(f"Best_top1_accuracy_epoch: {best_top1_epoch}.")
    print(f"Best_topk_accuracy: {best_topk}.")
    print(f"Best_topk_accuracy_epoch: {best_topk_epoch}.")
    print(f"Best_f1_score: {best_f1}.")
    print(f"Best_f1_score_epoch: {best_f1_epoch}.")
    print(f"Best_recall_score: {best_r}.")
    print(f"Best_recall_score_epoch: {best_r_epoch}.")
    print(f"Best_precision_score: {best_p}.")
    print(f"Best_precision_score_epoch: {best_p_epoch}.")
    print(f"Total time: {all_time}.")
    logging.info(f"Best_top1_accuracy: {best_top1}.")
    logging.info(f"Best_top1_accuracy_epoch: {best_top1_epoch}.")
    logging.info(f"Best_topk_accuracy: {best_topk}.")
    logging.info(f"Best_topk_accuracy_epoch: {best_topk_epoch}.")
    logging.info(f"Best_f1_score: {best_f1}.")
    logging.info(f"Best_f1_score_epoch: {best_f1_epoch}.")
    logging.info(f"Best_recall_score: {best_r}.")
    logging.info(f"Best_recall_score_epoch: {best_r_epoch}.")
    logging.info(f"Best_precision_score: {best_p}.")
    logging.info(f"Best_precision_score_epoch: {best_p_epoch}.")
    logging.info(f"Total time: {all_time}.")

    torch.save(best_top1_model, os.path.join(PATH, f"best_top1_model_{seed}.pth.tar"))
    torch.save(best_f1_model, os.path.join(PATH, f"best_f1_model_{seed}.pth.tar"))


def get_citrusdisease7_data_loaders(shuffle=False, batch_size=8):
    train_dataset = CitrusDisease7('./examples/data/citrusdisease7', pretrain=False, train=True,
                                   transform=transforms.Compose([transforms.ToTensor(), normalizer]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=2, drop_last=False, shuffle=shuffle, persistent_workers=True)

    val_dataset = CitrusDisease7('./examples/data/citrusdisease7', pretrain=False, train=False,
                                  transform=transforms.Compose([transforms.ToTensor(), normalizer]))

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             num_workers=2, drop_last=False, shuffle=shuffle, persistent_workers=True)
    return train_loader, val_loader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def precision_recall_f1_from_counts(TP, FP, FN, eps=1e-12):
    num_classes = len(TP)

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    p_mask = (TP + FP) > eps
    r_mask = (TP + FN) > eps

    precision[p_mask] = TP[p_mask] / (TP[p_mask] + FP[p_mask])
    recall[r_mask] = TP[r_mask] / (TP[r_mask] + FN[r_mask])

    f1_mask = (precision + recall) > eps
    f1[f1_mask] = 2 * precision[f1_mask] * recall[f1_mask] / (precision[f1_mask] + recall[f1_mask])

    return precision, recall, f1

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin-1')
    return dict

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        if name.split('.')[0] == 'feat_bn':
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


if __name__ == "__main__":
    eval()