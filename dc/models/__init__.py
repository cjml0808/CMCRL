from __future__ import absolute_import

from .resnet import *
from .resnet_ibn__ import *
# from .resnet_ibn import *
from .alexnet import *
from .densenet import *
from .mobilenetv2 import *

__factory = {
    'resnet18': resnet18, 'resnet18_bn': resnet18_bn, 'resnet18_4wa': resnet18_4wa,
    'resnet34': resnet34, 'resnet34_bn': resnet34_bn, 'resnet34_4wa': resnet34_4wa,
    'resnet50': resnet50, 'resnet50_bn': resnet50_bn, 'resnet50_4wa': resnet50_4wa,
    'resnet101': resnet101, 'resnet101_bn': resnet101_bn, 'resnet101_4wa': resnet101_4wa,
    'resnet152': resnet152, 'resnet152_bn': resnet152_bn, 'resnet152_4wa': resnet152_4wa,
    'resnet_ibn50a_ori': resnet_ibn50a_ori,
    # 'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn50a_bn': resnet_ibn50a_bn,
    'resnet_ibn50a_4h': resnet_ibn50a_4h,
    'resnet_ibn50a_4wa': resnet_ibn50a_4wa,
    'resnet_ibn101a': resnet_ibn101a,
    'alexnet': alexnet,
    'densenet121': densenet121,
    'mobilenetv2': mobilenetv2,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
