# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual_dense_block import RDB


# --- Downsampling block in GridDehazeNet  --- #
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


# --- Upsampling block in GridDehazeNet  --- #
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out


# --- Main model  --- #
class SIDNet(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(SIDNet, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)

        rdb_in_channels = depth_rate
        for i in range(height):  # i=0, 1, 2
            for j in range(width - 1):  # width = 0-6
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):  # i = 0, 1
            for j in range(width // 2):  # j = 0-3
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):  # 2 1 0
            for j in range(width // 2, width):  # j = 4 - 6
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride

        self.conv1 = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=2, dilation=2)
        self.conv3_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=4, dilation=4)
        self.conv4_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=8, dilation=8)
        self.conv5_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=16, dilation=16)
        self.conv6 = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.offset_conv1 = nn.Conv2d(depth_rate*4, depth_rate*2, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(depth_rate*2, depth_rate*4, 3, 1, 1, bias=True)
        self.upsamle1 = UpSample(depth_rate*4)
        self.upsamle2 = UpSample(depth_rate*2)

        self.rdb_2_1 = RDB(depth_rate * 2, num_dense_layer, growth_rate)
        self.rdb_1_1 = RDB(depth_rate, num_dense_layer, growth_rate)

    def forward(self, x):
        inp = self.conv_in(x)

        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)

        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])

        for i in range(1, self.height):  # i = 1, 2
            for j in range(1, self.width // 2):  # j = 1, 2, 3
                channel_num = int(2**(i-1)*self.stride*self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])

        x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
        k = j  # k = 3

        for j in range(self.width // 2 + 1, self.width):  # j = 5, 6, 7
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1])

        for i in range(self.height - 2, -1, -1):  # 2, 1, 0
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, k)](x_index[i][k]) + \
                              self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())

        for i in range(self.height - 2, -1, -1):  # i = 2, 1, 0
            for j in range(self.width // 2 + 1, self.width):  # j = 5, 6, 7
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())

        out = self.rdb_out(x_index[i][j])

        feat_extra = F.relu(self.conv1(x_index[-1][j]))
        feat_extra = F.relu(self.conv2_atrous(feat_extra))
        feat_extra = F.relu(self.conv3_atrous(feat_extra))
        feat_extra = F.relu(self.conv4_atrous(feat_extra))
        feat_extra = F.relu(self.conv5_atrous(feat_extra))
        feat_extra = F.relu(self.conv6(feat_extra))
        feat_extra = self.upsamle1(feat_extra, x_index[-2][j].size())
        feat_extra = self.coefficient[-2, 0, 0, :32][None, :, None, None] * x_index[-2][j] + self.coefficient[-2, 0, 0, 32:64][None, :, None, None] * feat_extra
        feat_extra = self.rdb_2_1(feat_extra)
        feat_extra = self.upsamle2(feat_extra, x_index[0][j].size())
        feat_extra = self.coefficient[0, 0, 0, :16][None, :, None, None] * out + self.coefficient[0, 0, 0, 16:32][None, :, None, None] * feat_extra
        out = self.rdb_1_1(feat_extra)
        out = F.relu(self.conv_out(out))
        # out = out + x
        return out, feat_extra
