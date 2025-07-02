from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer


__all__ = ['ResNet', 'resnet18', 'resnet18_bn', 'resnet18_4wa', 'resnet34', 'resnet34_bn', 'resnet34_4wa',
           'resnet50', 'resnet50_bn', 'resnet50_4wa', 'resnet101', 'resnet101_bn', 'resnet101_4wa',
           'resnet152', 'resnet152_bn', 'resnet152_4wa']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=True, dropout=0, num_classes=0, pooling_type='gem'):
        print('pooling_type: {}'.format(pooling_type))
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        if depth == '50' or depth == '101' or depth == '152':
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
        bs = x.size(0)
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False):
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


class ResNet_bn(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=True, dropout=0, num_classes=0, pooling_type='gem'):
        print('pooling_type: {}'.format(pooling_type))
        super(ResNet_bn, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet_bn.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet_bn.__factory[depth](pretrained=pretrained)
        if depth == '50' or depth == '101' or depth == '152':
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

        init.constant_(self.feat[1].weight, 1)
        init.constant_(self.feat[1].bias, 0)

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


class ResNet_4wa(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    layers = {
        18: [1, 1, 1, 1],
        34: [2, 3, 5, 2],
        50: [2, 3, 5, 2],
        101: [2, 3, 22, 2],
        152: [2, 7, 35, 2],
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=True, dropout=0, num_classes=0, pooling_type='gem'):
        print('pooling_type: {}'.format(pooling_type))
        super(ResNet_4wa, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet_4wa.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet_4wa.__factory[depth](pretrained=pretrained)
        if depth == '50' or depth == '101' or depth == '152':
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

            out_planes = ResNet_4wa.layers[depth]
            if depth == 18 or depth == 34:
                out_planes_1 = resnet.layer1[out_planes[0]].conv2.out_channels
                out_planes_2 = resnet.layer2[out_planes[1]].conv2.out_channels
                out_planes_3 = resnet.layer3[out_planes[2]].conv2.out_channels
                out_planes_4 = resnet.fc.in_features
            else:
                out_planes_1 = resnet.layer1[out_planes[0]].conv3.out_channels
                out_planes_2 = resnet.layer2[out_planes[1]].conv3.out_channels
                out_planes_3 = resnet.layer3[out_planes[2]].conv3.out_channels
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

            self.feat = nn.Linear(self.num_features * 4, self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)

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

        bn_x = torch.cat([bn_x_1, bn_x_2, bn_x_3, bn_x_4], dim=-1)
        bn_x = self.feat(bn_x)

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


def resnet18(**kwargs):
    return ResNet(18, **kwargs)

def resnet18_bn(**kwargs):
    return ResNet_bn(18, **kwargs)

def resnet18_4wa(**kwargs):
    return ResNet_4wa(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)

def resnet34_bn(**kwargs):
    return ResNet_bn(34, **kwargs)

def resnet34_4wa(**kwargs):
    return ResNet_4wa(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)

def resnet50_bn(**kwargs):
    return ResNet_bn(50, **kwargs)

def resnet50_4wa(**kwargs):
    return ResNet_4wa(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)

def resnet101_bn(**kwargs):
    return ResNet_bn(101, **kwargs)

def resnet101_4wa(**kwargs):
    return ResNet_4wa(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

def resnet152_bn(**kwargs):
    return ResNet_bn(152, **kwargs)

def resnet152_4wa(**kwargs):
    return ResNet_4wa(152, **kwargs)