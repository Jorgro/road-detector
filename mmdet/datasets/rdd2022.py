# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class RDD2022Dataset(XMLDataset):
    """Reader for the WIDER Face dataset in PASCAL VOC format.

    Conversion scripts can be found in
    https://github.com/sovrasov/wider-face-pascal-voc-annotations
    """
    CLASSES = ('D00', 'D10', 'D20', 'D40')

    PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192)]

    def __init__(self, **kwargs):
        super(RDD2022Dataset, self).__init__(**kwargs)
