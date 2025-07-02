from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init

import torch
from .pooling import build_pooling_layer

from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a


__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a', 'resnet_ibn50a_ori', 'resnet_ibn50a_bn', 'resnet_ibn50a_4wa',
           'resnet_ibn50a_4h']


class ResNetIBN_ori(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):

        print('pooling_type: {}'.format(pooling_type))
        super(ResNetIBN_ori, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN_ori.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):

        print('pooling_type: {}'.format(pooling_type))
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Sequential(nn.Linear(out_planes, out_planes * 2),
                                          nn.ReLU(), nn.Linear(out_planes * 2, self.num_features))
                init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
                init.constant_(self.feat[0].bias, 0)
                init.kaiming_normal_(self.feat[2].weight, mode='fan_out')
                init.constant_(self.feat[2].bias, 0)
                # self.feat = nn.Linear(out_planes, self.num_features)
                # self.feat_bn = nn.BatchNorm1d(self.num_features)
                # init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
                # init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            # self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        # init.constant_(self.feat_bn.weight, 1)
        # init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat(x)
            # bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        # if self.training is False:
        bn_x = F.normalize(bn_x)
            # return bn_x

        # if self.norm:
        #     bn_x = F.normalize(bn_x)
        # elif self.has_embedding:
        #     bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ResNetIBN_bn(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):

        print('pooling_type: {}'.format(pooling_type))
        super(ResNetIBN_bn, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN_bn.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Sequential(nn.Linear(out_planes, out_planes * 2), nn.BatchNorm1d(out_planes * 2),
                                          nn.ReLU(), nn.Linear(out_planes * 2, self.num_features))
                init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
                init.constant_(self.feat[0].bias, 0)
                init.kaiming_normal_(self.feat[3].weight, mode='fan_out')
                init.constant_(self.feat[3].bias, 0)
                # self.feat = nn.Linear(out_planes, self.num_features)
                # self.feat_bn = nn.BatchNorm1d(self.num_features)
                # init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
                # init.constant_(self.feat.bias, 0)
                init.constant_(self.feat[1].weight, 1)
                init.constant_(self.feat[1].bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            # self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat(x)
            # bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ResNetIBN_4h(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):

        print('pooling_type: {}'.format(pooling_type))
        super(ResNetIBN_4h, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN_4h.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base_1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.base_2 = resnet.layer2
        self.base_3 = resnet.layer3
        self.base_4 = resnet.layer4
        # self.base = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.num_classes = num_classes

            out_planes_1 = resnet.layer1[2].conv3.out_channels
            out_planes_2 = resnet.layer2[3].conv3.out_channels
            out_planes_3 = resnet.layer3[5].conv3.out_channels
            out_planes_4 = resnet.fc.in_features

            # Append new layers
            self.feat_1 = nn.Sequential(nn.Linear(out_planes_1, out_planes_1 * 2), nn.BatchNorm1d(out_planes_1 * 2),
                                        nn.ReLU(), nn.Linear(out_planes_1 * 2, self.num_features))
            init.kaiming_normal_(self.feat_1[0].weight, mode='fan_out')
            init.constant_(self.feat_1[0].bias, 0)
            init.kaiming_normal_(self.feat_1[3].weight, mode='fan_out')
            init.constant_(self.feat_1[3].bias, 0)
            init.constant_(self.feat_1[1].weight, 1)
            init.constant_(self.feat_1[1].bias, 0)
            self.feat_2 = nn.Sequential(nn.Linear(out_planes_2, out_planes_2 * 2), nn.BatchNorm1d(out_planes_2 * 2),
                                        nn.ReLU(), nn.Linear(out_planes_2 * 2, self.num_features))
            init.kaiming_normal_(self.feat_2[0].weight, mode='fan_out')
            init.constant_(self.feat_2[0].bias, 0)
            init.kaiming_normal_(self.feat_2[3].weight, mode='fan_out')
            init.constant_(self.feat_2[3].bias, 0)
            init.constant_(self.feat_2[1].weight, 1)
            init.constant_(self.feat_2[1].bias, 0)
            self.feat_3 = nn.Sequential(nn.Linear(out_planes_3, out_planes_3 * 2), nn.BatchNorm1d(out_planes_3 * 2),
                                        nn.ReLU(), nn.Linear(out_planes_3 * 2, self.num_features))
            init.kaiming_normal_(self.feat_3[0].weight, mode='fan_out')
            init.constant_(self.feat_3[0].bias, 0)
            init.kaiming_normal_(self.feat_3[3].weight, mode='fan_out')
            init.constant_(self.feat_3[3].bias, 0)
            init.constant_(self.feat_3[1].weight, 1)
            init.constant_(self.feat_3[1].bias, 0)
            self.feat_4 = nn.Sequential(nn.Linear(out_planes_4, out_planes_4 * 2), nn.BatchNorm1d(out_planes_4 * 2),
                                        nn.ReLU(), nn.Linear(out_planes_4 * 2, self.num_features))
            init.kaiming_normal_(self.feat_4[0].weight, mode='fan_out')
            init.constant_(self.feat_4[0].bias, 0)
            init.kaiming_normal_(self.feat_4[3].weight, mode='fan_out')
            init.constant_(self.feat_4[3].bias, 0)
            init.constant_(self.feat_4[1].weight, 1)
            init.constant_(self.feat_4[1].bias, 0)

            self.feat_1[1].bias.requires_grad_(False)
            self.feat_2[1].bias.requires_grad_(False)
            self.feat_3[1].bias.requires_grad_(False)
            self.feat_4[1].bias.requires_grad_(False)

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        x = self.base_1(x)
        bn_x_1 = self.gap(x)
        bn_x_1 = bn_x_1.view(bn_x_1.size(0), -1)
        bn_x_1 = self.feat_1(bn_x_1)

        x = self.base_2(x)
        bn_x_2 = self.gap(x)
        bn_x_2 = bn_x_2.view(bn_x_2.size(0), -1)
        bn_x_2 = self.feat_2(bn_x_2)

        x = self.base_3(x)
        bn_x_3 = self.gap(x)
        bn_x_3 = bn_x_3.view(bn_x_3.size(0), -1)
        bn_x_3 = self.feat_3(bn_x_3)

        x = self.base_4(x)
        bn_x_4 = self.gap(x)
        bn_x_4 = bn_x_4.view(bn_x_4.size(0), -1)
        bn_x_4 = self.feat_4(bn_x_4)

        if self.training is False:
            bn_x_1 = F.normalize(bn_x_1)
            bn_x_2 = F.normalize(bn_x_2)
            bn_x_3 = F.normalize(bn_x_3)
            bn_x_4 = F.normalize(bn_x_4)
            return [bn_x_1, bn_x_2, bn_x_3, bn_x_4]

        if self.norm:
            bn_x_4 = F.normalize(bn_x_4)

        if self.dropout > 0:
            bn_x_4 = self.drop(bn_x_4)

        if self.num_classes > 0:
            prob = self.classifier(bn_x_4)
        else:
            return [bn_x_1, bn_x_2, bn_x_3, bn_x_4]

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ResNetIBN_4wa(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):

        print('pooling_type: {}'.format(pooling_type))
        super(ResNetIBN_4wa, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN_4wa.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base_1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.base_2 = resnet.layer2
        self.base_3 = resnet.layer3
        self.base_4 = resnet.layer4
        # self.base = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes_1 = resnet.layer1[2].conv3.out_channels
            out_planes_2 = resnet.layer2[3].conv3.out_channels
            out_planes_3 = resnet.layer3[5].conv3.out_channels
            out_planes_4 = resnet.fc.in_features

            # Append new layers
            self.feat_1 = nn.Sequential(nn.Linear(out_planes_1, out_planes_1 * 2), nn.BatchNorm1d(out_planes_1 * 2),
                                        nn.ReLU(), nn.Linear(out_planes_1 * 2, self.num_features))
            init.kaiming_normal_(self.feat_1[0].weight, mode='fan_out')
            init.constant_(self.feat_1[0].bias, 0)
            init.kaiming_normal_(self.feat_1[3].weight, mode='fan_out')
            init.constant_(self.feat_1[3].bias, 0)
            init.constant_(self.feat_1[1].weight, 1)
            init.constant_(self.feat_1[1].bias, 0)
            self.feat_2 = nn.Sequential(nn.Linear(out_planes_2, out_planes_2 * 2), nn.BatchNorm1d(out_planes_2 * 2),
                                        nn.ReLU(), nn.Linear(out_planes_2 * 2, self.num_features))
            init.kaiming_normal_(self.feat_2[0].weight, mode='fan_out')
            init.constant_(self.feat_2[0].bias, 0)
            init.kaiming_normal_(self.feat_2[3].weight, mode='fan_out')
            init.constant_(self.feat_2[3].bias, 0)
            init.constant_(self.feat_2[1].weight, 1)
            init.constant_(self.feat_2[1].bias, 0)
            self.feat_3 = nn.Sequential(nn.Linear(out_planes_3, out_planes_3 * 2), nn.BatchNorm1d(out_planes_3 * 2),
                                        nn.ReLU(), nn.Linear(out_planes_3 * 2, self.num_features))
            init.kaiming_normal_(self.feat_3[0].weight, mode='fan_out')
            init.constant_(self.feat_3[0].bias, 0)
            init.kaiming_normal_(self.feat_3[3].weight, mode='fan_out')
            init.constant_(self.feat_3[3].bias, 0)
            init.constant_(self.feat_3[1].weight, 1)
            init.constant_(self.feat_3[1].bias, 0)
            if self.has_embedding:
                self.feat_4 = nn.Sequential(nn.Linear(out_planes_4, out_planes_4 * 2), nn.BatchNorm1d(out_planes_4 * 2),
                                            nn.ReLU(), nn.Linear(out_planes_4 * 2, self.num_features))
                init.kaiming_normal_(self.feat_4[0].weight, mode='fan_out')
                init.constant_(self.feat_4[0].bias, 0)
                init.kaiming_normal_(self.feat_4[3].weight, mode='fan_out')
                init.constant_(self.feat_4[3].bias, 0)
                init.constant_(self.feat_4[1].weight, 1)
                init.constant_(self.feat_4[1].bias, 0)

                self.feat_1[1].bias.requires_grad_(False)
                self.feat_2[1].bias.requires_grad_(False)
                self.feat_3[1].bias.requires_grad_(False)
                self.feat_4[1].bias.requires_grad_(False)

                self.feat = nn.Linear(self.num_features * 4, self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes_4
                self.feat_bn = nn.BatchNorm1d(self.num_features)

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        x = self.base_1(x)
        bn_x_1 = self.gap(x)
        bn_x_1 = bn_x_1.view(bn_x_1.size(0), -1)
        bn_x_1 = self.feat_1(bn_x_1)

        x = self.base_2(x)
        bn_x_2 = self.gap(x)
        bn_x_2 = bn_x_2.view(bn_x_2.size(0), -1)
        bn_x_2 = self.feat_2(bn_x_2)

        x = self.base_3(x)
        bn_x_3 = self.gap(x)
        bn_x_3 = bn_x_3.view(bn_x_3.size(0), -1)
        bn_x_3 = self.feat_3(bn_x_3)

        x = self.base_4(x)
        bn_x_4 = self.gap(x)
        bn_x_4 = bn_x_4.view(bn_x_4.size(0), -1)
        if self.has_embedding:
            bn_x_4 = self.feat_4(bn_x_4)
            bn_x = torch.cat([bn_x_1, bn_x_2, bn_x_3, bn_x_4], dim=-1)
            bn_x = self.feat(bn_x)
        else:
            bn_x = self.feat_bn(bn_x_4)

        # bn_x_4 = self.feat_4(bn_x_4)
        #
        # bn_x = torch.cat([bn_x_1, bn_x_2, bn_x_3, bn_x_4], dim=-1)
        # bn_x = self.feat(bn_x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


# class ResNetIBN_4wa(nn.Module):
#     __factory = {
#         '50a': resnet50_ibn_a,
#         '101a': resnet101_ibn_a
#     }
#
#     def __init__(self, depth, pretrained=True, cut_at_pooling=False,
#                  num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):
#
#         print('pooling_type: {}'.format(pooling_type))
#         super(ResNetIBN_4wa, self).__init__()
#
#         self.depth = depth
#         self.pretrained = pretrained
#         self.cut_at_pooling = cut_at_pooling
#
#         resnet = ResNetIBN_4wa.__factory[depth](pretrained=pretrained)
#         resnet.layer4[0].conv2.stride = (1, 1)
#         resnet.layer4[0].downsample[0].stride = (1, 1)
#
#         self.base_1 = nn.Sequential(
#             resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
#         self.base_2 = resnet.layer2
#         self.base_3 = resnet.layer3
#         self.base_4 = resnet.layer4
#         # self.base = nn.Sequential(
#         #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
#         #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
#
#         self.gap = build_pooling_layer(pooling_type)
#
#         if not self.cut_at_pooling:
#             self.num_features = num_features
#             self.norm = norm
#             self.dropout = dropout
#             self.num_classes = num_classes
#
#             out_planes_1 = resnet.layer1[2].conv3.out_channels
#             out_planes_2 = resnet.layer2[3].conv3.out_channels
#             out_planes_3 = resnet.layer3[5].conv3.out_channels
#             out_planes_4 = resnet.fc.in_features
#
#             # Append new layers
#             self.feat_1 = nn.Sequential(nn.Linear(out_planes_1, out_planes_1 * 2), nn.BatchNorm1d(out_planes_1 * 2),
#                                         nn.ReLU(), nn.Linear(out_planes_1 * 2, self.num_features))
#             init.kaiming_normal_(self.feat_1[0].weight, mode='fan_out')
#             init.constant_(self.feat_1[0].bias, 0)
#             init.kaiming_normal_(self.feat_1[3].weight, mode='fan_out')
#             init.constant_(self.feat_1[3].bias, 0)
#             init.constant_(self.feat_1[1].weight, 1)
#             init.constant_(self.feat_1[1].bias, 0)
#             self.feat_2 = nn.Sequential(nn.Linear(out_planes_2, out_planes_2 * 2), nn.BatchNorm1d(out_planes_2 * 2),
#                                         nn.ReLU(), nn.Linear(out_planes_2 * 2, self.num_features))
#             init.kaiming_normal_(self.feat_2[0].weight, mode='fan_out')
#             init.constant_(self.feat_2[0].bias, 0)
#             init.kaiming_normal_(self.feat_2[3].weight, mode='fan_out')
#             init.constant_(self.feat_2[3].bias, 0)
#             init.constant_(self.feat_2[1].weight, 1)
#             init.constant_(self.feat_2[1].bias, 0)
#             self.feat_3 = nn.Sequential(nn.Linear(out_planes_3, out_planes_3 * 2), nn.BatchNorm1d(out_planes_3 * 2),
#                                         nn.ReLU(), nn.Linear(out_planes_3 * 2, self.num_features))
#             init.kaiming_normal_(self.feat_3[0].weight, mode='fan_out')
#             init.constant_(self.feat_3[0].bias, 0)
#             init.kaiming_normal_(self.feat_3[3].weight, mode='fan_out')
#             init.constant_(self.feat_3[3].bias, 0)
#             init.constant_(self.feat_3[1].weight, 1)
#             init.constant_(self.feat_3[1].bias, 0)
#             self.feat_4 = nn.Sequential(nn.Linear(out_planes_4, out_planes_4 * 2), nn.BatchNorm1d(out_planes_4 * 2),
#                                         nn.ReLU(), nn.Linear(out_planes_4 * 2, self.num_features))
#             init.kaiming_normal_(self.feat_4[0].weight, mode='fan_out')
#             init.constant_(self.feat_4[0].bias, 0)
#             init.kaiming_normal_(self.feat_4[3].weight, mode='fan_out')
#             init.constant_(self.feat_4[3].bias, 0)
#             init.constant_(self.feat_4[1].weight, 1)
#             init.constant_(self.feat_4[1].bias, 0)
#
#             self.feat_1[1].bias.requires_grad_(False)
#             self.feat_2[1].bias.requires_grad_(False)
#             self.feat_3[1].bias.requires_grad_(False)
#             self.feat_4[1].bias.requires_grad_(False)
#
#             self.feat = nn.Linear(self.num_features * 4, self.num_features)
#             init.kaiming_normal_(self.feat.weight, mode='fan_out')
#             init.constant_(self.feat.bias, 0)
#
#             if self.dropout > 0:
#                 self.drop = nn.Dropout(self.dropout)
#             if self.num_classes > 0:
#                 self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
#                 init.normal_(self.classifier.weight, std=0.001)
#
#         if not pretrained:
#             self.reset_params()
#
#     def forward(self, x):
#         x = self.base_1(x)
#         bn_x_1 = self.gap(x)
#         bn_x_1 = bn_x_1.view(bn_x_1.size(0), -1)
#         bn_x_1 = self.feat_1(bn_x_1)
#
#         x = self.base_2(x)
#         bn_x_2 = self.gap(x)
#         bn_x_2 = bn_x_2.view(bn_x_2.size(0), -1)
#         bn_x_2 = self.feat_2(bn_x_2)
#
#         x = self.base_3(x)
#         bn_x_3 = self.gap(x)
#         bn_x_3 = bn_x_3.view(bn_x_3.size(0), -1)
#         bn_x_3 = self.feat_3(bn_x_3)
#
#         x = self.base_4(x)
#         bn_x_4 = self.gap(x)
#         bn_x_4 = bn_x_4.view(bn_x_4.size(0), -1)
#         bn_x_4 = self.feat_4(bn_x_4)
#
#         bn_x = torch.cat([bn_x_1, bn_x_2, bn_x_3, bn_x_4], dim=-1)
#         bn_x = self.feat(bn_x)
#
#         if self.training is False:
#             bn_x = F.normalize(bn_x)
#             return bn_x
#
#         if self.norm:
#             bn_x = F.normalize(bn_x)
#
#         if self.dropout > 0:
#             bn_x = self.drop(bn_x)
#
#         if self.num_classes > 0:
#             prob = self.classifier(bn_x)
#         else:
#             return bn_x
#
#         return prob
#
#     def reset_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

# class ResNetIBN_bn(nn.Module):
#     __factory = {
#         '50a': resnet50_ibn_a,
#         '101a': resnet101_ibn_a
#     }
#
#     def __init__(self, depth, pretrained=True, cut_at_pooling=False,
#                  num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):
#
#         print('pooling_type: {}'.format(pooling_type))
#         super(ResNetIBN_bn, self).__init__()
#
#         self.depth = depth
#         self.pretrained = pretrained
#         self.cut_at_pooling = cut_at_pooling
#
#         resnet = ResNetIBN_bn.__factory[depth](pretrained=pretrained)
#         resnet.layer4[0].conv2.stride = (1, 1)
#         resnet.layer4[0].downsample[0].stride = (1, 1)
#
#         self.base = nn.Sequential(
#             resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
#             resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
#
#         self.gap = build_pooling_layer(pooling_type)
#
#         if not self.cut_at_pooling:
#             self.num_features = num_features
#             self.norm = norm
#             self.dropout = dropout
#             self.has_embedding = num_features > 0
#             self.num_classes = num_classes
#
#             out_planes = resnet.fc.in_features
#
#             # Append new layers
#             if self.has_embedding:
#                 self.feat = nn.Sequential(nn.Linear(out_planes, out_planes * 2), nn.BatchNorm1d(out_planes * 2),
#                                           nn.ReLU(), nn.Linear(out_planes * 2, self.num_features))
#                 init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
#                 init.constant_(self.feat[0].bias, 0)
#                 init.kaiming_normal_(self.feat[3].weight, mode='fan_out')
#                 init.constant_(self.feat[3].bias, 0)
#                 # self.feat = nn.Linear(out_planes, self.num_features)
#                 # self.feat_bn = nn.BatchNorm1d(self.num_features)
#                 # init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
#                 # init.constant_(self.feat.bias, 0)
#             else:
#                 # Change the num_features to CNN output channels
#                 self.num_features = out_planes
#                 self.feat_bn = nn.BatchNorm1d(self.num_features)
#             # self.feat_bn.bias.requires_grad_(False)
#             if self.dropout > 0:
#                 self.drop = nn.Dropout(self.dropout)
#             if self.num_classes > 0:
#                 self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
#                 init.normal_(self.classifier.weight, std=0.001)
#
#         init.constant_(self.feat[1].weight, 1)
#         init.constant_(self.feat[1].bias, 0)
#
#         if not pretrained:
#             self.reset_params()
#
#     def forward(self, x):
#         x = self.base(x)
#
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#
#         if self.cut_at_pooling:
#             return x
#
#         if self.has_embedding:
#             bn_x = self.feat(x)
#             # bn_x = self.feat_bn(self.feat(x))
#         else:
#             bn_x = self.feat_bn(x)
#
#         # if self.training is False:
#         bn_x = F.normalize(bn_x)
#             # return bn_x
#
#         # if self.norm:
#         #     bn_x = F.normalize(bn_x)
#         # elif self.has_embedding:
#         #     bn_x = F.relu(bn_x)
#
#         if self.dropout > 0:
#             bn_x = self.drop(bn_x)
#
#         if self.num_classes > 0:
#             prob = self.classifier(bn_x)
#         else:
#             return bn_x
#
#         return prob
#
#     def reset_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

# class ResNetIBN(nn.Module):
#     __factory = {
#         '50a': resnet50_ibn_a,
#         '101a': resnet101_ibn_a
#     }
#
#     def __init__(self, depth, pretrained=True, cut_at_pooling=False,
#                  num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):
#
#         print('pooling_type: {}'.format(pooling_type))
#         super(ResNetIBN, self).__init__()
#
#         self.depth = depth
#         self.pretrained = pretrained
#         self.cut_at_pooling = cut_at_pooling
#
#         resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
#         resnet.layer4[0].conv2.stride = (1, 1)
#         resnet.layer4[0].downsample[0].stride = (1, 1)
#
#         self.base = nn.Sequential(
#             resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
#             resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
#
#         self.gap = build_pooling_layer(pooling_type)
#
#         if not self.cut_at_pooling:
#             self.num_features = num_features
#             self.norm = norm
#             self.dropout = dropout
#             self.has_embedding = num_features > 0
#             self.num_classes = num_classes
#
#             out_planes = resnet.fc.in_features
#
#             # Append new layers
#             if self.has_embedding:
#                 self.feat = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(self.num_features),
#                                           nn.ReLU(), nn.Linear(out_planes, out_planes))
#                 init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
#                 init.constant_(self.feat[0].bias, 0)
#                 init.kaiming_normal_(self.feat[3].weight, mode='fan_out')
#                 init.constant_(self.feat[3].bias, 0)
#                 # self.feat = nn.Linear(out_planes, self.num_features)
#                 # self.feat_bn = nn.BatchNorm1d(self.num_features)
#                 # init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
#                 # init.constant_(self.feat.bias, 0)
#             else:
#                 # Change the num_features to CNN output channels
#                 self.num_features = out_planes
#                 self.feat_bn = nn.BatchNorm1d(self.num_features)
#             self.feat[1].bias.requires_grad_(False)
#             # self.feat_bn.bias.requires_grad_(False)
#             if self.dropout > 0:
#                 self.drop = nn.Dropout(self.dropout)
#             if self.num_classes > 0:
#                 self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
#                 init.normal_(self.classifier.weight, std=0.001)
#
#         init.constant_(self.feat[1].weight, 1)
#         init.constant_(self.feat[1].bias, 0)
#         # init.constant_(self.feat_bn.weight, 1)
#         # init.constant_(self.feat_bn.bias, 0)
#
#         if not pretrained:
#             self.reset_params()
#
#     def forward(self, x):
#         x = self.base(x)
#
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#
#         if self.cut_at_pooling:
#             return x
#
#         if self.has_embedding:
#             bn_x = self.feat(x)
#             # bn_x = self.feat_bn(self.feat(x))
#         else:
#             bn_x = self.feat_bn(x)
#
#         if self.training is False:
#             bn_x = F.normalize(bn_x)
#             return bn_x
#
#         if self.norm:
#             bn_x = F.normalize(bn_x)
#         elif self.has_embedding:
#             bn_x = F.relu(bn_x)
#
#         if self.dropout > 0:
#             bn_x = self.drop(bn_x)
#
#         if self.num_classes > 0:
#             prob = self.classifier(bn_x)
#         else:
#             return bn_x
#
#         return prob
#
#     def reset_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

def resnet_ibn50a_ori(**kwargs):
    return ResNetIBN_ori('50a', **kwargs)

def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)

def resnet_ibn50a_bn(**kwargs):
    return ResNetIBN_bn('50a', **kwargs)

def resnet_ibn50a_4h(**kwargs):
    return ResNetIBN_4h('50a', **kwargs)

def resnet_ibn50a_4wa(**kwargs):
    return ResNetIBN_4wa('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)










class ResNetIBN_3(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):

        print('pooling_type: {}'.format(pooling_type))
        super(ResNetIBN_3, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN_3.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Sequential(nn.Linear(out_planes, out_planes * 2),
                                          nn.ReLU(), nn.Linear(out_planes * 2, out_planes * 2),
                                          nn.ReLU(), nn.Linear(out_planes * 2, self.num_features))
                init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
                init.constant_(self.feat[0].bias, 0)
                init.kaiming_normal_(self.feat[2].weight, mode='fan_out')
                init.constant_(self.feat[2].bias, 0)
                init.kaiming_normal_(self.feat[4].weight, mode='fan_out')
                init.constant_(self.feat[4].bias, 0)
                # self.feat = nn.Linear(out_planes, self.num_features)
                # self.feat_bn = nn.BatchNorm1d(self.num_features)
                # init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
                # init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            # self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        # init.constant_(self.feat_bn.weight, 1)
        # init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat(x)
            # bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        # if self.training is False:
        bn_x = F.normalize(bn_x)
            # return bn_x

        # if self.norm:
        #     bn_x = F.normalize(bn_x)
        # elif self.has_embedding:
        #     bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ResNetIBN_2bn(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='gem'):

        print('pooling_type: {}'.format(pooling_type))
        super(ResNetIBN_2bn, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN_2bn.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Sequential(nn.Linear(out_planes, out_planes * 2), nn.BatchNorm1d(out_planes * 2),
                                          nn.ReLU(), nn.Linear(out_planes * 2, self.num_features), nn.BatchNorm1d(self.num_features))
                init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
                init.constant_(self.feat[0].bias, 0)
                init.kaiming_normal_(self.feat[3].weight, mode='fan_out')
                init.constant_(self.feat[3].bias, 0)
                init.constant_(self.feat[1].weight, 1)
                init.constant_(self.feat[1].bias, 0)
                init.constant_(self.feat[4].weight, 1)
                init.constant_(self.feat[4].bias, 0)
                # self.feat = nn.Linear(out_planes, self.num_features)
                # self.feat_bn = nn.BatchNorm1d(self.num_features)
                # init.kaiming_normal_(self.feat[0].weight, mode='fan_out')
                # init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            # self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat(x)
            # bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        # if self.training is False:
        bn_x = F.normalize(bn_x)
            # return bn_x

        # if self.norm:
        #     bn_x = F.normalize(bn_x)
        # elif self.has_embedding:
        #     bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)



def resnet_ibn50a_3(**kwargs):
    return ResNetIBN_3('50a', **kwargs)

def resnet_ibn50a_2bn(**kwargs):
    return ResNetIBN_2bn('50a', **kwargs)
