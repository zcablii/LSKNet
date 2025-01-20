# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from ...builder import ROTATED_HEADS
from .rotated_bbox_head import RotatedBBoxHead
from .reg_block import StripBlock

@ROTATED_HEADS.register_module()
class StripHead_(RotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg

    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_xy_wh_convs=0,
                 num_reg_xy_wh_fcs=0,
                 num_reg_theta_convs=0,
                 num_reg_theta_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(StripHead_, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        # assert (num_shared_convs + num_shared_fcs + num_cls_convs +
        #         num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        # if num_cls_convs > 0 or num_reg_convs > 0:
        #     assert num_shared_fcs == 0
        # if not self.with_cls:
        #     assert num_cls_convs == 0 and num_cls_fcs == 0
        # if not self.with_reg:
        #     assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_xy_wh_convs = num_reg_xy_wh_convs
        self.num_reg_xy_wh_fcs = num_reg_xy_wh_fcs

        self.num_reg_theta_convs = num_reg_theta_convs
        self.num_reg_theta_fcs = num_reg_theta_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg xy specific branch
        self.reg_xy_wh_convs, self.reg_xy_wh_fcs, self.reg_xy_wh_last_dim = \
            self._add_conv_strip_fc_branch(
                self.num_reg_xy_wh_convs, self.num_reg_xy_wh_fcs, self.shared_out_channels)
            
        # add reg theta specific branch
        self.reg_theta_convs, self.reg_theta_fcs, self.reg_theta_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_theta_convs, self.num_reg_theta_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_xy_wh_fcs == 0:
                self.reg_xy_wh_last_dim *= self.roi_feat_area
            if self.num_reg_theta_fcs == 0:
                self.reg_theta_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg_xy_wh = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            out_dim_reg_theta = (1 if self.reg_class_agnostic else 1 *
                           self.num_classes)
            self.fc_reg_xy_wh = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_xy_wh_last_dim,
                out_features=out_dim_reg_xy_wh)
            self.fc_reg_theta = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_theta_last_dim,
                out_features=out_dim_reg_theta)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_xy_wh_fcs'),
                        dict(name='reg_theta_fcs')
                    ])
            ]

    def _add_conv_strip_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))       
                branch_convs.append(
                    StripBlock(self.conv_out_channels)
                )
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
            
        x_reg_xy_wh = x_reg
        for conv in self.reg_xy_wh_convs:
            x_reg_xy_wh = conv(x_reg_xy_wh)
        
        # print(x_reg_xy_wh.shape)
        if x_reg_xy_wh.dim() > 2:
            if self.with_avg_pool:
                x_reg_xy_wh = self.avg_pool(x_reg_xy_wh)
            x_reg_xy_wh = x_reg_xy_wh.flatten(1)
            print(x_reg_xy_wh.shape)
        for fc in self.reg_xy_wh_fcs:
            x_reg_xy_wh = self.relu(fc(x_reg_xy_wh))
        # print(x_reg_xy_wh.shape)

        x_reg_theta = x_reg
        for conv in self.reg_theta_convs:
            x_reg_theta = conv(x_reg_theta)
        if x_reg_theta.dim() > 2:
            if self.with_avg_pool:
                x_reg_theta = self.avg_pool(x_reg_theta)
            x_reg_theta = x_reg_theta.flatten(1)
        for fc in self.reg_theta_fcs:
            x_reg_theta = self.relu(fc(x_reg_theta))
            
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        xy_wh_pred = self.fc_reg_xy_wh(x_reg_xy_wh) if self.with_reg else None
        theta_pred = self.fc_reg_theta(x_reg_theta) if self.with_reg else None
        bbox_pred = torch.cat((xy_wh_pred, theta_pred), dim=1)
        return cls_score, bbox_pred

@ROTATED_HEADS.register_module()
class StripHead(StripHead_):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(StripHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_xy_wh_convs=1,
            num_reg_xy_wh_fcs=0,
            num_reg_theta_convs=0,
            num_reg_theta_fcs=2,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)