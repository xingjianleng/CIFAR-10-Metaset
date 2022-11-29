# Based on (heavily modified) the implementation from
# https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/models/ResNet.py

import torch
import torch.nn.functional as F
from torch import nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(num_features=planes)

    def forward(self, x):
        residual = x 
        # NOTE: In original implementation, calculation has the order BN -> ReLU -> Conv
        #       which is clearly wrong. Changed to Conv -> BN -> ReLU as stated in original paper.
        residual = self.conv1(residual)
        residual = self.bn1(residual)
        residual = F.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)

        if self.downsample is not None:
            x = self.downsample(x)
        
        # NOTE: In original implementation, ReLU was called for the residual output.
        #       However, in their paper ReLU was used for (x + residual)
        return F.relu(x + residual)


class Downsample(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(Downsample, self).__init__()
        # NOTE: It was wrong in the original implementation that the
        #       size of AvgPool kernel size equals stride. 
        #       Rather, it should be constantly 1 as shown in FaceBook implementation
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)
        assert out_planes % in_planes == 0
        self.expand_ratio = out_planes // in_planes

    def forward(self, x):
        x = self.avg(x)
        # In original FaceBook Lua implementation, dim=2 because index starts with 1 in Lua
        return torch.cat([x] + [torch.zeros_like(x)] * (self.expand_ratio - 1), dim=1)


class ResNetCifar(nn.Module):
    def __init__(self, depth, classes=10, channels=3):
        assert (depth - 2) % 6 == 0         # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResNetCifar, self).__init__()

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=16)
        self.in_planes = 16
        self.layer1 = self._make_layer(16, stride=1)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(64, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, classes)

        # initialization
        for m in self.modules():
            # kaiming initialization at https://arxiv.org/abs/1502.01852
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                
    def _make_layer(self, planes, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = Downsample(self.in_planes, planes, stride)
        layers = [BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(self.N - 1):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # NOTE: It was wrong in the implementation that BatchNorm and ReLU were put after all layers
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
