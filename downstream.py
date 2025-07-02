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
from sklearn.metrics import f1_score, recall_score, precision_score

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
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def eval(mode='full_finetune', e=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    num_class = 7
    model = models.create('resnet_ibn50a_ori', num_features=2048, norm=True, dropout=0,
                          num_classes=num_class, pooling_type='gem')
    checkpoint_exist = True
    PATH = f"E:/Code/densityclustering-master/copy_of_server/densityclustering-master/examples/logs/" \
           f"citrus_disease_{num_class}_resnet_ibn50a/citrus_disease_7_(w.o.PadRHF)_1"
    if checkpoint_exist:
        checkpoint = torch.load(os.path.join(PATH, f"model_{e}.pth.tar"), map_location=device)
        state_dict = checkpoint['state_dict']
        model = copy_state_dict(state_dict, model, strip='module.')

    if mode == 'final_finetune':
        logging.basicConfig(filename=os.path.join(PATH, f"2048_final_testing_{checkpoint['epoch']}_normalize.log"),
                            level=logging.DEBUG)
        logging.info("Using device:", device)
        logging.info(f"Loading model from the folder {PATH.split('/')[-1]}.")
        # logging.info(f"Pretraining top1: {checkpoint['top1']}.")
        # logging.info(f"Pretraining epoch: {checkpoint['epoch']}.")

        if num_class == 7:
            k = 5
            train_loader, test_loader = get_citrusdisease7_data_loaders()
        elif num_class == 4:
            k = 3
            train_loader, test_loader = get_citrusdisease4_data_loaders()
        print(f"Dataset: citrusdisease{num_class}")
        logging.info(f"Dataset: citrusdisease{num_class}")

        # freeze all layers but the last fc
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
        logging.info(f"Start SimCLR testing for {epochs} epochs.")
        logging.info(f"Testing with dataset: citrusdisease{num_class}.")
        begin = time.time()
        for epoch in range(epochs):
            epoch_begin = time.time()
            top1_train_accuracy = 0
            f1_train, r_train, p_train = 0, 0, 0
            for counter, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                model.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                # 这里的logits的维度很可能会大于下游任务的标签维数，但是模型在训练过程中学习如何在这个高维空间中表示和区分不同的类别，
                # 帮助模型将fc层的输出映射到与下游任务相匹配的概率分布上去。此时，输出空间的某些维度会与特定类别相关，而其他维度则变得不那么重要了
                top1 = accuracy(logits, y_batch, topk=(1,))
                f1, r, p = f1rp(logits, y_batch)
                top1_train_accuracy += top1[0]
                f1_train += f1
                r_train += r
                p_train += p

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            top1_train_accuracy /= (counter + 1)
            f1_train /= (counter + 1)
            r_train /= (counter + 1)
            p_train /= (counter + 1)
            top1_accuracy = 0
            topk_accuracy = 0
            f1_test, r_test, p_test = 0, 0, 0
            for counter, (x_batch, y_batch) in enumerate(test_loader):
                if x_batch.shape[0] > 1:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                elif x_batch.shape[0] == 1:
                    x_batch = torch.cat([x_batch, x_batch], dim=0).to(device)
                    y_batch = torch.cat([y_batch, y_batch], dim=0).to(device)

                logits = model(x_batch)

                top1, topk = accuracy(logits, y_batch, topk=(1, k))
                f1, r, p = f1rp(logits, y_batch)
                top1_accuracy += top1[0]
                topk_accuracy += topk[0]
                f1_test += f1
                r_test += r
                p_test += p

            top1_accuracy /= (counter + 1)
            topk_accuracy /= (counter + 1)
            f1_test /= (counter + 1)
            r_test /= (counter + 1)
            p_test /= (counter + 1)

            if epoch == 0 or top1_accuracy > best_top1:
                best_top1 = top1_accuracy
                best_top1_epoch = epoch
            if epoch == 0 or topk_accuracy > best_topk:
                best_topk = topk_accuracy
                best_topk_epoch = epoch
            if epoch == 0 or f1_test > best_f1:
                best_f1 = f1_test
                best_f1_epoch = epoch
            if epoch == 0 or r_test > best_r:
                best_r = r_test
                best_r_epoch = epoch
            if epoch == 0 or p_test > best_p:
                best_p = p_test
                best_p_epoch = epoch
            epoch_time = time.time() - epoch_begin

            print(
                f"Epoch {epoch}\tLoss: {loss}\tTop1 Train accuracy {top1_train_accuracy.item()}"
                f"\tF1 train score {f1_train}\tRecall train score {r_train}\tPrecision train score {p_train}"
                f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTopk test acc: {topk_accuracy.item()}"
                f"\tF1 test score: {f1_test}\tRecall test score: {r_test}\tPrecision test score: {p_test}"
                f"\tEpoch time:{epoch_time}")
            logging.info(
                f"Epoch {epoch}\tLoss: {loss}\tTop1 Train accuracy {top1_train_accuracy.item()}"
                f"\tF1 train score {f1_train}\tRecall train score {r_train}\tPrecision train score {p_train}"
                f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTopk test acc: {topk_accuracy.item()}"
                f"\tF1 test score: {f1_test}\tRecall test score: {r_test}\tPrecision test score: {p_test}"
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
    elif mode == 'full_finetune':
        logging.basicConfig(filename=os.path.join(PATH, f"2048_full_testing_{checkpoint['epoch']}_normalize.log"),
                            level=logging.DEBUG)
        logging.info("Using device:", device)
        logging.info(f"Loading model from the folder {PATH.split('/')[-1]}.")
        # logging.info(f"Pretraining top1: {checkpoint['top1']}.")
        # logging.info(f"Pretraining epoch: {checkpoint['epoch']}.")

        train_loader, test_loader = get_citrusdisease7_data_loaders()
        print("Dataset: citrusdisease7")
        logging.info(f"Dataset: citrusdisease7")

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        epochs = 300
        best_top1 = 0
        best_top5 = 0
        best_top1_epoch = 0
        best_top5_epoch = 0
        best_f1, best_r, best_p = 0, 0, 0
        best_f1_epoch, best_r_epoch, best_p_epoch = 0, 0, 0
        logging.info(f"Start SimCLR testing for {epochs} epochs.")
        logging.info(f"Testing with dataset: citrusdisease7.")
        begin = time.time()
        for epoch in range(epochs):
            epoch_begin = time.time()
            top1_train_accuracy = 0
            f1_train, r_train, p_train = 0, 0, 0
            for counter, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                model.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                # 这里的logits的维度很可能会大于下游任务的标签维数，但是模型在训练过程中学习如何在这个高维空间中表示和区分不同的类别，
                # 帮助模型将fc层的输出映射到与下游任务相匹配的概率分布上去。此时，输出空间的某些维度会与特定类别相关，而其他维度则变得不那么重要了
                top1 = accuracy(logits, y_batch, topk=(1,))
                f1, r, p = f1rp(logits, y_batch)
                top1_train_accuracy += top1[0]
                f1_train += f1
                r_train += r
                p_train += p

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            top1_train_accuracy /= (counter + 1)
            f1_train /= (counter + 1)
            r_train /= (counter + 1)
            p_train /= (counter + 1)
            top1_accuracy = 0
            top5_accuracy = 0
            f1_test, r_test, p_test = 0, 0, 0
            for counter, (x_batch, y_batch) in enumerate(test_loader):
                if x_batch.shape[0] > 1:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                elif x_batch.shape[0] == 1:
                    x_batch = torch.cat([x_batch, x_batch], dim=0).to(device)
                    y_batch = torch.cat([y_batch, y_batch], dim=0).to(device)

                logits = model(x_batch)

                top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
                f1, r, p = f1rp(logits, y_batch)
                top1_accuracy += top1[0]
                top5_accuracy += top5[0]
                f1_test += f1
                r_test += r
                p_test += p

            top1_accuracy /= (counter + 1)
            top5_accuracy /= (counter + 1)
            f1_test /= (counter + 1)
            r_test /= (counter + 1)
            p_test /= (counter + 1)

            if epoch == 0 or top1_accuracy > best_top1:
                best_top1 = top1_accuracy
                best_top1_epoch = epoch
            if epoch == 0 or top5_accuracy > best_top5:
                best_top5 = top5_accuracy
                best_top5_epoch = epoch
            if epoch == 0 or f1_test > best_f1:
                best_f1 = f1_test
                best_f1_epoch = epoch
            if epoch == 0 or r_test > best_r:
                best_r = r_test
                best_r_epoch = epoch
            if epoch == 0 or p_test > best_p:
                best_p = p_test
                best_p_epoch = epoch
            epoch_time = time.time() - epoch_begin

            print(
                f"Epoch {epoch}\tLoss: {loss}\tTop1 Train accuracy {top1_train_accuracy.item()}"
                f"\tF1 train score {f1_train}\tRecall train score {r_train}\tPrecision train score {p_train}"
                f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}"
                f"\tF1 test score: {f1_test}\tRecall test score: {r_test}\tPrecision test score: {p_test}"
                f"\tEpoch time:{epoch_time}")
            logging.info(
                f"Epoch {epoch}\tLoss: {loss}\tTop1 Train accuracy {top1_train_accuracy.item()}"
                f"\tF1 train score {f1_train}\tRecall train score {r_train}\tPrecision train score {p_train}"
                f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}"
                f"\tF1 test score: {f1_test}\tRecall test score: {r_test}\tPrecision test score: {p_test}"
                f"\tEpoch time:{epoch_time}")
        all_time = time.time() - begin
        print(f"Best_top1_accuracy: {best_top1}.")
        print(f"Best_top1_accuracy_epoch: {best_top1_epoch}.")
        print(f"Best_top5_accuracy: {best_top5}.")
        print(f"Best_top5_accuracy_epoch: {best_top5_epoch}.")
        print(f"Best_f1_score: {best_f1}.")
        print(f"Best_f1_score_epoch: {best_f1_epoch}.")
        print(f"Best_recall_score: {best_r}.")
        print(f"Best_recall_score_epoch: {best_r_epoch}.")
        print(f"Best_precision_score: {best_p}.")
        print(f"Best_precision_score_epoch: {best_p_epoch}.")
        print(f"Total time: {all_time}.")
        logging.info(f"Best_top1_accuracy: {best_top1}.")
        logging.info(f"Best_top1_accuracy_epoch: {best_top1_epoch}.")
        logging.info(f"Best_top5_accuracy: {best_top5}.")
        logging.info(f"Best_top5_accuracy_epoch: {best_top5_epoch}.")
        logging.info(f"Best_f1_score: {best_f1}.")
        logging.info(f"Best_f1_score_epoch: {best_f1_epoch}.")
        logging.info(f"Best_recall_score: {best_r}.")
        logging.info(f"Best_recall_score_epoch: {best_r_epoch}.")
        logging.info(f"Best_precision_score: {best_p}.")
        logging.info(f"Best_precision_score_epoch: {best_p_epoch}.")
        logging.info(f"Total time: {all_time}.")


def get_citrusdisease7_data_loaders(shuffle=False, batch_size=8):
    train_dataset = CitrusDisease7('./examples/data/citrusdisease7', pretrain=False, train=True,
                                   transform=transforms.Compose([transforms.ToTensor(), normalizer]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=2, drop_last=False, shuffle=shuffle, persistent_workers=True)

    test_dataset = CitrusDisease7('./examples/data/citrusdisease7', pretrain=False, train=False,
                                  transform=transforms.Compose([transforms.ToTensor(), normalizer]))

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=2, drop_last=False, shuffle=shuffle, persistent_workers=True)
    return train_loader, test_loader

def get_citrusdisease4_data_loaders(shuffle=False, batch_size=8):
    train_dataset = CitrusDisease4('./examples/data/citrusdisease4', pretrain=False, train=True,
                                   transform=transforms.Compose([transforms.ToTensor(), normalizer]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=2, drop_last=False, shuffle=shuffle, persistent_workers=True)

    test_dataset = CitrusDisease4('./examples/data/citrusdisease4', pretrain=False, train=False,
                                  transform=transforms.Compose([transforms.ToTensor(), normalizer]))

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=2, drop_last=False, shuffle=shuffle, persistent_workers=True)
    return train_loader, test_loader

def get_citrusdisease6_data_loaders(shuffle=False, batch_size=8):
    train_dataset = CitrusDisease6('./examples/data/citrusdisease6', pretrain=False, train=True,
                                   transform=transforms.Compose([transforms.ToTensor(), normalizer]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=2, drop_last=False, shuffle=shuffle, persistent_workers=True)

    test_dataset = CitrusDisease6('./examples/data/citrusdisease6', pretrain=False, train=False,
                                  transform=transforms.Compose([transforms.ToTensor(), normalizer]))

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=2, drop_last=False, shuffle=shuffle, persistent_workers=True)
    return train_loader, test_loader

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

def f1rp(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        f1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro') * 100
        r = recall_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro') * 100
        p = precision_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro') * 100
        return f1, r, p

def randomized_svd(image, k, q=0, p=0):
    # Step 1: Sample column space of image with P matrix
    ny = image.shape[1]
    P = np.random.randn(ny, k+p)
    # image = image.cpu().detach().numpy()
    Z = image @ P
    for i in range(q):
        Z = image @ (image.T @ Z)

    Q, R = np.linalg.qr(Z, mode='reduced')

    # Step 2: Compute SVD on projected Y = Q.T @ image
    Y = Q.T @ image
    U, S, V = np.linalg.svd(Y, full_matrices=False)  # 分解原图像
    U = Q @ U
    S = np.diag(S)
    # 对原图像的矩阵进行压缩
    # compressed_U = U[:, :k]
    # compressed_S = np.diag(S[:k])
    # compressed_V = V[:k, :]
    # compressed_image = np.dot(U, np.dot(S, V))  # 对矩阵进行乘法运算，恢复压缩图像
    return U, S, V

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
    # PATH = os.getcwd() + '/runs/Jan09_20-25-57_LAPTOP-IVD0GQ6G/'
    # PATH = os.getcwd() + '/runs/final_simclr/cifar10_RSVD_resnet18_256_128_5(2)_100_240630-701/_Jun30_19-31-47_LAPTOP-IVD0GQ6G'
    for e in [10, 5, 4, 3, 2, 1]:
        eval(mode='final_finetune', e=e)

    # content = unpickle('./datasets/cifar-10-batches-py/data_batch_1')
    #
    # print(content)