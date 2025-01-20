import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ...builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer
from mmcv.ops import DeformConv2d


class StripBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.strip_conv1 = nn.Conv2d(dim,dim,kernel_size=(1, 19), stride=1, padding=(0, 9), groups=dim)
        self.strip_conv2 = nn.Conv2d(dim,dim,kernel_size=(19, 1), stride=1, padding=(9, 0), groups=dim)     
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.strip_conv1(attn)
        attn = self.strip_conv2(attn)
        attn = self.conv1(attn)

        return u * attn