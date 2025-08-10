
import torch
import math
from torch import nn
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import dyhead._C as _C

class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1
    ):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        _C.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _C.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out

modulated_deform_conv = ModulatedDeformConvFunction.apply

class ModulatedDeformConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True
    ):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels,
            in_channels // groups,
            *self.kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return modulated_deform_conv(
            input, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups)

    def __repr__(self):
        return "".join([
            "{}(".format(self.__class__.__name__),
            "in_channels={}, ".format(self.in_channels),
            "out_channels={}, ".format(self.out_channels),
            "kernel_size={}, ".format(self.kernel_size),
            "stride={}, ".format(self.stride),
            "dilation={}, ".format(self.dilation),
            "padding={}, ".format(self.padding),
            "groups={}, ".format(self.groups),
            "deformable_groups={}, ".format(self.deformable_groups),
            "bias={})".format(self.with_bias),
        ])

class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConvPack, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, deformable_groups, bias)

        self.conv_offset_mask = nn.Conv2d(
            self.in_channels // self.groups,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(
            input, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups)


import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DYReLU(nn.Module):
    def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
                 init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super(DYReLU, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        # print('init_a: {}, init_b: {}'.format(self.init_a, self.init_b))

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid()
        )
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:  # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)

        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True)/3
            out = out * ys

        return out
    

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling.backbone import Backbone

from .deform import ModulatedDeformConv
from .dyrelu import h_sigmoid, DYReLU


class Conv3x3Norm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv3x3Norm, self).__init__()

        self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x


class DyConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True))

        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(in_channels, out_channels)
        self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        next_x = {}
        feature_names = list(x.keys())
        for level, name in enumerate(feature_names):

            feature = x[name]

            offset_mask = self.offset(feature)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, 18:, :, :].sigmoid()
            conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]
            if level > 0:
                temp_fea.append(self.DyConv[2](x[feature_names[level - 1]], **conv_args))
            if level < len(x) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](x[feature_names[level + 1]], **conv_args),
                                                    size=[feature.size(2), feature.size(3)]))
            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))

            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
            mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
            next_x[name] = self.relu(mean_fea)

        return next_x


class DyHead(Backbone):
    def __init__(self, cfg, backbone):
        super(DyHead, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        channels = cfg.MODEL.DYHEAD.CHANNELS

        dyhead_tower = []
        for i in range(cfg.MODEL.DYHEAD.NUM_CONVS):
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=Conv3x3Norm,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self._out_feature_strides = self.backbone._out_feature_strides
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: channels for k in self._out_features}
        self._size_divisibility = list(self._out_feature_strides.values())[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        x = self.backbone(x)
        dyhead_tower = self.dyhead_tower(x)
        return dyhead_tower
