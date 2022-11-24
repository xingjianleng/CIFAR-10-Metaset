import torch
import torch.nn as nn
import torch.nn.functional as F


# Modified from TorchVision implementation of ResNet for ImageNet
class BasicBlock(nn.Module):
    expansion = 1

    # BasicBlock in ResNet, containing two convolution layers
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample=None,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    # The ResNet implementation for CIFAR-10 dataset
    def __init__(
        self,
        n: int,
    ):
        super().__init__()

        num_classes = 10  # as defined in CIFAR-10
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        # no subsampling in the first layer
        self.layer1 = self._make_layer(n, 16, 1)
        self.layer2 = self._make_layer(n, 32, 2)
        self.layer3 = self._make_layer(n, 64, 2)
        # output from layer3 has size (64 * 8 * 8)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        # output from pooling layer has size (64 * 1 * 1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, blocks, out_channels, stride):
        # no matter how many blocks there are, this layer will half the feature size
        # by setting stride=2, which performs subsampling
        downsample = None
        layers = []

        # the first block may not maintain the shape, e.g., in_channels != out_channels, stride != 1
        # as suggested in the paper, use 1*1 convolution to transform the matrix size (option B)
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers.append(
            BasicBlock(
                in_channels=self.in_channels, out_channels=out_channels, stride=stride, downsample=downsample
            )
        )

        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            # these layers maintain the shape, provided stride=1
            layers.append(
                BasicBlock(
                    in_channels=self.in_channels, out_channels=out_channels
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet110(device):
    return ResNet(n=18).to(device=device)
