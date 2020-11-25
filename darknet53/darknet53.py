import torch
import torch.nn as nn
from layers import *
from collections import OrderedDict
import torch.backends.cudnn as cudnn

class DarkNet53(nn.Module):
    def __init__(self, num_classes):
        super(DarkNet53, self).__init__()

        input_channels = 32
        stage_cfg = {'stage_2':1, 'stage_3':2, 'stage_4':8, 'stage_5':8, 'stage_6':4}

        # Network
        self.stage1 = Conv2dBatchLeaky(3, input_channels, 3, 1, 1)
        self.stage2 = Stage(input_channels, stage_cfg['stage_2'])
        self.stage3 = Stage(input_channels*(2**1), stage_cfg['stage_3'])
        self.stage4 = Stage(input_channels*(2**2), stage_cfg['stage_4'])
        self.stage5 = Stage(input_channels*(2**3), stage_cfg['stage_5'])
        self.stage6 = Stage(input_channels*(2**4), stage_cfg['stage_6'])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        stage6 = self.stage6(stage5)

        x = self.avgpool(stage6)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    darknet = DarkNet53(num_classes=10)
    with torch.no_grad():
        darknet.eval()
        data = torch.rand(1,3,256,256)
        try:
            print(darknet(data))
        except Exception as e:
            print(e)



    #print(darknet)
    # print("============================")
    # for m in darknet.modules():
    #     print(m)



