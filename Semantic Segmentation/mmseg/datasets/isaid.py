# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict

@DATASETS.register_module()
class isAIDDataset(CustomDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    MAPPING = OrderedDict({
    'background': (0, 0, 0),
    'ship': (0, 0, 63),
    'storage_tank': (0, 191, 127),
    'baseball_diamond': (0, 63, 0),
    'tennis_court': (0, 63, 127),
    'basketball_court': (0, 63, 191),
    'ground_Track_Field': (0, 63, 255),
    'bridge': (0, 127, 63),
    'large_Vehicle': (0, 127, 127),
    'small_Vehicle': (0, 0, 127),
    'helicopter': (0, 0, 191),
    'swimming_pool': (0, 0, 255),
    'roundabout': (0, 63, 63),
    'soccer_ball_field': (0, 127, 191),
    'plane': (0, 127, 255),
    'harbor': (0, 100, 155),
    })
    
    CLASSES = tuple(MAPPING.keys())

    PALETTE = list(map(lambda x:list(x),list(MAPPING.values())))

    def __init__(self, **kwargs):
        super(isAIDDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_instance_color_RGB.png',
            reduce_zero_label=False,
            ignore_index=255,
            **kwargs)
