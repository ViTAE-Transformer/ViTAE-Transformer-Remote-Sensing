from functools import partial
from pyexpat import model
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
from torch.nn.functional import instance_norm
from torch.nn.modules.batchnorm import BatchNorm2d
from .NormalCell import NormalCell
from .ReductionCell import ReductionCell

#from ..custom_load import load_checkpoint
from mmdet.utils import get_root_logger
#from mmcv.utils.registry import BACKBONES
from ...builder import BACKBONES

import warnings
import torch.nn.functional as F
from collections import OrderedDict
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
import math

from mmcv_custom import load_checkpoint

from torch.nn.modules.batchnorm import _BatchNorm

#from torch.nn.modules.batchnorm import _LazyNormBase

class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48, img_size=None):
        self.img_size = img_size
        self.inter_channel = inter_channel
        self.out_channel = out_channels
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        return x

    def flops(self, ) -> float:
        flops = 0
        flops += 3 * self.inter_channel * self.img_size[0] * self.img_size[1] // 4 * 9
        flops += self.img_size[0] * self.img_size[1] // 4 * self.inter_channel
        flops += self.inter_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16 * 9
        flops += self.img_size[0] * self.img_size[1] // 16 * self.out_channel
        flops += self.out_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16
        return flops

class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7, RC_heads=1, NC_heads=6, dilations=[1, 2, 3, 4],
                RC_op='cat', RC_tokens_type='performer', NC_tokens_type='transformer', RC_group=1, NC_group=64, NC_depth=2, dpr=0.1, mlp_ratio=4., qkv_bias=True, 
                qk_scale=None, drop=0, attn_drop=0., norm_layer=nn.LayerNorm, class_token=False, gamma=False, init_values=1e-4, SE=False, window_size=7, relative_pos=False):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = self.img_size // self.downsample_ratios
        self.RC_kernel_size = kernel_size
        self.RC_heads = RC_heads
        self.NC_heads = NC_heads
        self.dilations = dilations
        self.RC_op = RC_op
        self.RC_tokens_type = RC_tokens_type
        self.RC_group = RC_group
        self.NC_group = NC_group
        self.NC_depth = NC_depth
        self.relative_pos = relative_pos
        if RC_tokens_type == 'stem':
            # 直接用俩3*3 stride=2的卷积作下采样
            self.RC = PatchEmbedding(inter_channel=token_dims//2, out_channels=token_dims, img_size=img_size)
        elif downsample_ratios > 1:
            self.RC = ReductionCell(img_size, in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
                            RC_heads, dilations, op=RC_op, tokens_type=RC_tokens_type, group=RC_group, gamma=gamma, init_values=init_values, SE=SE, relative_pos=relative_pos, window_size=window_size)
        else:
            self.RC = nn.Identity()
        self.NC = nn.ModuleList([
            NormalCell(token_dims, NC_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                       drop_path=dpr[i] if isinstance(dpr, list) else dpr, norm_layer=norm_layer, class_token=class_token, group=NC_group, tokens_type=NC_tokens_type,
                       gamma=gamma, init_values=init_values, SE=SE, img_size=img_size // downsample_ratios, window_size=window_size, shift_size=0, relative_pos=relative_pos)
        for i in range(NC_depth)])

    def forward(self, x, H, W):
        # 每个layer先过下采样block，然后跟上多个normel block
        x, H, W = self.RC(x, H, W)
        for nc in self.NC:
            x, H, W = nc(x, H, W)
        return x, H, W

@BACKBONES.register_module()
class ViTAE_Window_NoShift_basic(nn.Module):
    def __init__(self, img_size=224, in_chans=3, stages=4, embed_dims=64, token_dims=64, downsample_ratios=[4, 2, 2, 2], kernel_size=[7, 3, 3, 3], 
                RC_heads=[1, 1, 1, 1], NC_heads=4, dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
                RC_op='cat', RC_tokens_type=['performer', 'transformer', 'transformer', 'transformer'], NC_tokens_type='transformer',
                RC_group=[1, 1, 1, 1], NC_group=[1, 32, 64, 64], NC_depth=[2, 2, 6, 2], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000,
                gamma=False, init_values=1e-4, SE=False, window_size=7, relative_pos=False, pretrained=None, init_cfg=None,
                frozen_stages=-1,norm_eval=True):

        # assert not (init_cfg and pretrained), \
        #     'init_cfg and pretrained cannot be specified at the same time'
        # if isinstance(pretrained, str):
        #     warnings.warn('DeprecationWarning: pretrained is deprecated, '
        #                   'please use "init_cfg" instead')
        #     init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        # elif pretrained is None:
        #     init_cfg = init_cfg
        # else:
        #     raise TypeError('pretrained must be a str or None')

        super(ViTAE_Window_NoShift_basic, self).__init__()

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.init_cfg = init_cfg

        self.num_classes = num_classes
        self.stages = stages
        repeatOrNot = (lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)])
        self.embed_dims = repeatOrNot(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in range(stages)]
        self.downsample_ratios = repeatOrNot(downsample_ratios, stages)
        self.kernel_size = repeatOrNot(kernel_size, stages)
        self.RC_heads = repeatOrNot(RC_heads, stages)
        self.NC_heads = repeatOrNot(NC_heads, stages)
        self.dilaions = repeatOrNot(dilations, stages)
        self.RC_op = repeatOrNot(RC_op, stages)
        self.RC_tokens_type = repeatOrNot(RC_tokens_type, stages)
        self.NC_tokens_type = repeatOrNot(NC_tokens_type, stages)
        self.RC_group = repeatOrNot(RC_group, stages)
        self.NC_group = repeatOrNot(NC_group, stages)
        self.NC_depth = repeatOrNot(NC_depth, stages)
        self.mlp_ratio = repeatOrNot(mlp_ratio, stages)
        self.qkv_bias = repeatOrNot(qkv_bias, stages)
        self.qk_scale = repeatOrNot(qk_scale, stages)
        self.drop = repeatOrNot(drop_rate, stages)
        self.attn_drop = repeatOrNot(attn_drop_rate, stages)
        self.norm_layer = repeatOrNot(norm_layer, stages)
        self.relative_pos = relative_pos

        self.pos_drop = nn.Dropout(p=drop_rate)
        depth = np.sum(self.NC_depth)
        # 生成长度为block总数，从0到 drop_path_rate的等差数列
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i==0 else self.NC_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i], self.NC_depth[i], dpr[startDpr:self.NC_depth[i]+startDpr],
                mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i], drop=self.drop[i], attn_drop=self.attn_drop[i],
                norm_layer=self.norm_layer[i], gamma=gamma, init_values=init_values, SE=SE, window_size=window_size, relative_pos=relative_pos)
            )
            img_size = img_size // self.downsample_ratios[i] # 每个layer的输入尺寸
            in_chans = self.tokens_dims[i]
        self.layers = nn.ModuleList(Layers)

        # Classifier head
        #self.head = nn.Linear(self.tokens_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        #self.apply(self._init_weights)

        # add a norm layer for each output

        out_indices=(0, 1, 2, 3)

        for i_layer in out_indices:
            layer = norm_layer(self.tokens_dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self, pretrained=None):

        if pretrained != None:

            logger = get_root_logger()

            ckpt = _load_checkpoint(
                    pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'state_dict_ema' in ckpt:
                _state_dict = ckpt['state_dict_ema']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # delete relative_position_index since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete relative_coords_table since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete attn_mask since we always re-init it
            attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
            for k in attn_mask_keys:
                del state_dict[k]

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
            
            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)
        
    def _freeze_stages(self):
        for i in range(0, self.frozen_stages):
            for param in self.layers[i].parameters():
                param.requires_grad = False

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'cls_token'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, Wh, Ww):
        outs = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, Wh, Ww = layer(x, Wh, Ww)
            b, n, _ = x.shape
            #wh = int(math.sqrt(n))
            #norm_layer = getattr(self, f'norm{i}')
            #x_out = norm_layer(x)
            outs.append(x.view(b, Wh, Ww, -1).permute(0, 3, 1, 2).contiguous())

        return outs

    def forward(self, x):

        Wh, Ww = x.size(2), x.size(3)

        x = self.forward_features(x, Wh, Ww)
        #x = self.head(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed"""
        super(ViTAE_Window_NoShift_basic, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
    
    # def train(self, mode=True, tag='default'):
    #     r"""Sets the module in training mode.

    #     This has any effect only on certain modules. See documentations of
    #     particular modules for details of their behaviors in training/evaluation
    #     mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    #     etc.

    #     Args:
    #         mode (bool): whether to set training mode (``True``) or evaluation
    #                      mode (``False``). Default: ``True``.

    #     Returns:
    #         Module: self
    #     """
    #     self.training = mode
    #     if tag == 'default':
    #         for module in self.children():
    #             module.train(mode)
    #     elif tag == 'linear':
    #         for module in self.children():
    #             module.eval()
    #         self.head.train()
    #     elif tag == 'linearLN':
    #         for module in self.children():
    #             module.train(False, tag=tag)
    #         self.head.train()
    #     return self

    # def train(self, mode=True, tag='default'):
    #     self.training = mode
    #     if tag == 'default':
    #         for module in self.modules():
    #             if module.__class__.__name__ != 'ViTAE_Window_NoShift_basic':
    #                 module.train(mode)
    #     elif tag == 'linear':
    #         for module in self.modules():
    #             if module.__class__.__name__ != 'ViTAE_Window_NoShift_basic':
    #                 module.eval()
    #                 for param in module.parameters():
    #                     param.requires_grad = False
    #     elif tag == 'linearLNBN':
    #         for module in self.modules():
    #             if module.__class__.__name__ != 'ViTAE_Window_NoShift_basic':
    #                 if isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm2d):
    #                     module.train(mode)
    #                     for param in module.parameters():
    #                         param.requires_grad = True
    #                 else:
    #                     module.eval()
    #                     for param in module.parameters():
    #                         param.requires_grad = False
    #     self.head.train(mode)
    #     for param in self.head.parameters():
    #         param.requires_grad = True
    #     return self