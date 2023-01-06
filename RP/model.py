from torch import nn
import torch.nn.functional as F

from ResNet.model import BasicBlock, Downsample


class ResNetRotation(nn.Module):
    def __init__(self, depth, classes=10, channels=3):
        assert (depth - 2) % 6 == 0         # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResNetRotation, self).__init__()

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=16)
        self.in_planes = 16
        self.layer1 = self._make_layer(16, stride=1)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(64, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc_classification = nn.Linear(64, classes)
        self.fc_rotation = nn.Linear(64, 4)  # rotation invariance

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
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc_classification(x), self.fc_rotation(x)
