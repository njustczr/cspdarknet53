import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBatchLeaky(nn.Module):
    """
    This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation='leaky', leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(k/2) for k in kernel_size]
        else:
            self.padding = int(kernel_size/2)
        self.leaky_slope = leaky_slope

        # Layer
        if activation == "leaky":
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.LeakyReLU(self.leaky_slope, inplace=True)
            )
        elif activation == "linear":
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False)
            )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x

class SmallBlock(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dBatchLeaky(nchannels, nchannels, 1, 1),
            Conv2dBatchLeaky(nchannels, nchannels, 3, 1)
        )
        self.active_linear = Conv2dBatchLeaky(nchannels, nchannels, 1, 1, activation='linear')

    def forward(self, data):
        short_cut = data + self.features(data)
        active_linear = self.active_linear(short_cut)

        return active_linear

# Stage1  conv [256,256,3]->[256,256,32]

class Stage2(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        self.conv1 = Conv2dBatchLeaky(nchannels, 2*nchannels, 3, 2)
        self.conv2 = Conv2dBatchLeaky(2*nchannels, 2*nchannels, 1, 1)
        self.conv3 = Conv2dBatchLeaky(2*nchannels, nchannels, 1, 1)
        self.conv4 = Conv2dBatchLeaky(nchannels, 2*nchannels, 3, 1)

        self.active_linear = Conv2dBatchLeaky(2*nchannels, 2*nchannels, 1, 1, activation='linear')
        self.conv5_l = Conv2dBatchLeaky(2*nchannels, 2*nchannels, 1, 1)
        self.conv5_r = Conv2dBatchLeaky(2*nchannels, 2*nchannels, 1, 1)


    def forward(self, data):
        conv1 = self.conv1(data)
        route1 = conv1
        conv2 = self.conv2(route1)
        conv2_bk = conv2
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        shortcut = conv2_bk + conv4
        active_linear = self.active_linear(shortcut)
        conv5_r = self.conv5_r(active_linear)
        conv5_l = self.conv5_l(conv1)
        route2 = torch.cat([conv5_l, conv5_r], dim=1)

        return route2

class Stage3(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        self.conv1 = Conv2dBatchLeaky(nchannels, int(nchannels/2), 1, 1)
        self.conv2 = Conv2dBatchLeaky(int(nchannels/2), nchannels, 3, 2)

        self.conv3 = Conv2dBatchLeaky(nchannels, int(nchannels/2), 1, 1)
        self.block1 = SmallBlock(int(nchannels/2))
        self.block2 = SmallBlock(int(nchannels/2))

        self.conv4_l = Conv2dBatchLeaky(nchannels, int(nchannels/2), 1, 1)
        self.conv4_r = Conv2dBatchLeaky(int(nchannels/2), int(nchannels/2), 1, 1)

    def forward(self, data):
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)
        route1 = conv2
        conv3 = self.conv3(route1)
        block1 = self.block1(conv3)
        block2 = self.block2(block1)
        conv4_r = self.conv4_r(block2)
        conv4_l = self.conv4_l(conv2)
        route2 = torch.cat([conv4_l, conv4_r], dim=1)

        return route2

# Stage4 Stage5 Stage6
class Stage(nn.Module):
    def __init__(self, nchannels, nblocks):
        super().__init__()
        self.conv1 = Conv2dBatchLeaky(nchannels, nchannels, 1, 1)
        self.conv2 = Conv2dBatchLeaky(nchannels, 2*nchannels, 3, 2)
        self.conv3 = Conv2dBatchLeaky(2*nchannels, nchannels, 1, 1)
        blocks = []
        for i in range(nblocks):
            blocks.append(SmallBlock(nchannels))
        self.blocks = nn.Sequential(*blocks)
        self.conv4_l = Conv2dBatchLeaky(2*nchannels, nchannels, 1, 1)
        self.conv4_r = Conv2dBatchLeaky(nchannels, nchannels, 1, 1)

    def forward(self,data):
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)
        route1 = conv2
        conv3 = self.conv3(route1)
        blocks = self.blocks(conv3)
        conv4_r = self.conv4_r(blocks)
        conv4_l = self.conv4_l(conv2)
        route2 = torch.cat([conv4_l, conv4_r], dim=1)

        return route2







