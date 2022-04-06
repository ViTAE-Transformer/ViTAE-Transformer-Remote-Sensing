# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
from math import gamma
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
# from timm.models.registry import register_model
import numpy as np
from .base_model import ViTAE_Window_NoShift_basic

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ViTAE_stages3_7': _cfg(),
}

# @register_model
def ViTAE_Window_NoShift_12_basic_stages4_14(pretrained=False, **kwargs): # adopt performer for tokens to token
    # if pretrained:
        # kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = ViTAE_Window_NoShift_basic(RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model