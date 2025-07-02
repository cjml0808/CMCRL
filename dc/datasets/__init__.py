# Copyright (c) 2020-present.
# All rights reserved.
#
# Date: 2021-07-14
# Author: Jiabao Wang
# Email: jiabao_1108@163.com
#

from __future__ import absolute_import
import warnings

from .CUB_200_2011 import CUB_200_2011
from .FGVC_Aircraft import FGVC_Aircraft
from .Stanford_Dogs import Stanford_Dogs
from .Stanford_Cars import Stanford_Cars
from .Oxford_Flowers import Oxford_Flowers
from .Oxford_Pets import Oxford_Pets
from .Citrus_disease_6 import Citrus_disease_6
# from .Citrus_disease_6_finegrained import Citrus_disease_6
from .Citrus_disease_7 import Citrus_disease_7
from .Citrus_disease_4 import Citrus_disease_4
from .CDD import CDD


__factory = {
    'cub200': CUB_200_2011,
    'aircraft': FGVC_Aircraft,
    'dogs': Stanford_Dogs,
    'cars': Stanford_Cars,
    'flowers': Oxford_Flowers,
    'pets': Oxford_Pets,
    'citrusdisease6': Citrus_disease_6,
    'citrusdisease7': Citrus_disease_7,
    'citrusdisease4': Citrus_disease_4,
    'cdd': CDD,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. 
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
