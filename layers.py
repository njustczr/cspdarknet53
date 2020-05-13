import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBatchLeaky(nn.Module):
    """
    This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
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
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x

class StageBlock(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dBatchLeaky(nchannels, int(nchannels/2), 1, 1),
            Conv2dBatchLeaky(int(nchannels/2), nchannels, 3, 1)
        )

    def forward(self, data):
        return data + self.features(data)


class Stage(nn.Module):

    def __init__(self, nchannels, nblocks, stride=2):
        super().__init__()
        blocks = []
        blocks.append(Conv2dBatchLeaky(nchannels, 2*nchannels, 3, stride))
        for i in range(nblocks):
            blocks.append(StageBlock(2*nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)




