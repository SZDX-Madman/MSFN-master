import torch
import torch.nn as nn
import sys
import numpy as np

from torch.nn.parameter import Parameter
import torch.nn.functional as F
################
# Basic blocks
################


class upsample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(upsample, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_ch,in_ch*2,kernel_size=3,padding=1,padding_mode='reflect'),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(in_ch * 2, out_ch, 3, stride=2, padding=1, output_padding=1, )
                                )
    def forward(self,x):
        return self.conv(x)

class downsample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(downsample, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch//2, kernel_size=3,stride=2, padding=1, padding_mode='reflect'),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(in_ch//2, out_ch, 3, stride=1, padding=1,padding_mode='reflect' )
                                  )

    def forward(self, x):
        return self.conv(x)


# class sa_layer(nn.Module):
#     """Constructs a Channel Spatial Group module.
#     Args:
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self, channel, groups=64):
#         super(sa_layer, self).__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
#         self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
#
#         self.sigmoid = nn.Sigmoid()
#         self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
#
#     @staticmethod
#     def channel_shuffle(x, groups):
#         b, c, h, w = x.shape
#
#         x = x.reshape(b, groups, -1, h, w)
#         x = x.permute(0, 2, 1, 3, 4)
#
#         # flatten
#         x = x.reshape(b, -1, h, w)
#
#         return x
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         x = x.reshape(b * self.groups, -1, h, w)
#         x_0, x_1 = x.chunk(2, dim=1)
#
#         # channel attention
#         xn = self.avg_pool(x_0)
#         xn = self.cweight * xn + self.cbias
#         xn = x_0 * self.sigmoid(xn)
#
#         # spatial attention
#         xs = self.gn(x_1)
#         xs = self.sweight * xs + self.sbias
#         xs = x_1 * self.sigmoid(xs)
#
#         # concatenate along channel axis
#         out = torch.cat([xn, xs], dim=1)
#         out = out.reshape(b, -1, h, w)
#
#         out = self.channel_shuffle(out, 2)
#         return out
#
# class StripPooling(nn.Module):
#     """
#     Reference:
#     """
#     def __init__(self, in_channels, factor, up_kwargs):
#         super(StripPooling, self).__init__()
#         self.pool1_1 = nn.AdaptiveAvgPool2d((factor, None))
#         self.pool1_2 = nn.AdaptiveAvgPool2d((None, factor))
#
#         # self.conv0 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
#         self.conv2_1 = nn.Conv2d(in_channels, in_channels, (1, 3), 1, (0, 1), bias=False)
#
#         self.conv2_2 = nn.Conv2d(in_channels, in_channels, (3, 1), 1, (1, 0), bias=False)
#         # bilinear interpolate options
#         self._up_kwargs = up_kwargs
#
#     def forward(self, x):
#         _, _, h, w = x.size()
#         # f=self.conv0(x)
#         f1 = F.interpolate(self.conv2_1(self.pool1_1(x)), (h, w), **self._up_kwargs)
#         f2 = F.interpolate(self.conv2_2(self.pool1_2(x)), (h, w), **self._up_kwargs)
#         return f1+f2
#
#
# class MKSP(nn.Module):
#     def __init__(self,in_channels,up_kwargs):
#         super(MKSP, self).__init__()
#         self.sp1=StripPooling(in_channels//2,1,up_kwargs)
#         self.sp2=StripPooling(in_channels//2,3,up_kwargs)
#         self.sp3 = StripPooling(in_channels//2, 5,up_kwargs)
#         self.sp4 = StripPooling(in_channels//2, 7,up_kwargs)
#         self.conv0=nn.Conv2d(in_channels,in_channels//2,3,1,1)
#         self.conv1=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,stride=1,padding=1)
#         self.conv2=nn.Conv2d(in_channels,in_channels,3,1,1)
#         self.relu_=nn.ReLU(inplace=True)
#         self.sigmod_=nn.Sigmoid()
#     def forward(self,x):
#         f=self.conv0(x)
#         f1=self.sp1(f)
#         f2=self.sp2(f)
#         f3=self.sp3(f)
#         f4=self.sp4(f)
#         f=self.relu_(self.conv1(torch.cat([f1,f2,f3,f4],dim=1)))
#         f=self.sigmod_(self.conv2(f))
#         return x*f
# class AR(nn.Module):
#     def __init__(self,in_channels):
#         super(AR, self).__init__()
#         self.att=nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,1),
#                                nn.ReLU(inplace=True),
#                                nn.Conv2d(in_channels,in_channels,3,1,1),
#                                nn.Sigmoid())
#     def forward(self,x):
#         return x*self.att(x)
#
# class BA(nn.Module):
#     def __init__(self,in_channels,up_kwargs):
#         super(BA, self).__init__()
#         self.mksp=MKSP(in_channels,up_kwargs)
#         self.ar=AR(in_channels)
#     def forward(self,x):
#         return self.ar(self.mksp(x))