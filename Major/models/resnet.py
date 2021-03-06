

"""
Reference:
Yerlan Idelbayev: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""


import torch.nn as nn
import torch.nn.functional as F
from .base import AdversarialDefensiveModule
from .layerops import Sequential



__all__ = ["ResNet", "resnet8", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet1202"]

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, 3, 
                    stride=stride, padding=padding, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)



class BasicBlock(AdversarialDefensiveModule):

    def __init__(
        self, in_channels, out_channels,
        stride=1, shortcut=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.shortway = shortcut

    def forward(self, x):
        temp = self.activation(self.bn1(self.conv1(x)))
        outs = self.bn2(self.conv2(temp))
        outs2 = x if self.shortway is None else self.shortway(x)
        return self.activation(outs + outs2)


class ResNet(AdversarialDefensiveModule):

    def __init__(
        self, layers, num_classes=10, strides=(1,2,2),
        block=BasicBlock
    ):
        super(ResNet, self).__init__()

        if isinstance(strides, str):
            strides = list(map(int, strides))

        self.conv0 = conv3x3(3, 16)
        self.bn0 = nn.BatchNorm2d(16)
        self.cur_channels = 16

        self.layer1 = self._make_layer(block, 16, layers[0], strides[0]) # 16 x 32 x 32
        self.layer2 = self._make_layer(block, 32, layers[1], strides[1]) # 32 x 16 x 16
        self.layer3 = self._make_layer(block, 64, layers[2], strides[2]) # 64 x 8 x 8

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def _make_layer(self, block, out_channels, num_blocks, stride):

        shortcut = None
        if stride != 1 or out_channels != self.cur_channels:
            shortcut = conv1x1(self.cur_channels, out_channels, stride)
        
        layers = [block(self.cur_channels, out_channels, stride, shortcut)]
        self.cur_channels = out_channels
        for _ in range(num_blocks-1):
            layers.append(block(out_channels, out_channels))
        
        return Sequential(*layers)

    def forward(self, inputs):
        x = F.relu(self.bn0(self.conv0(inputs)))
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        features = self.avg_pool(l3).flatten(start_dim=1)
        outs = self.fc(features)
        return outs


def resnet8(num_classes=10, **kwargs):
    return ResNet([1, 1, 1], num_classes=num_classes, **kwargs)


def resnet20(num_classes=10, **kwargs):
    return ResNet([3, 3, 3], num_classes=num_classes, **kwargs)


def resnet32(num_classes=10, **kwargs):
    return ResNet([5, 5, 5], num_classes=num_classes, **kwargs)


def resnet44(num_classes=10, **kwargs):
    return ResNet([7, 7, 7], num_classes=num_classes, **kwargs)


def resnet56(num_classes=10, **kwargs):
    return ResNet([9, 9, 9], num_classes=num_classes, **kwargs)


def resnet110(num_classes=10, **kwargs):
    return ResNet([18, 18, 18], num_classes=num_classes, **kwargs)


def resnet1202(num_classes=10, **kwargs):
    return ResNet([200, 200, 200], num_classes=num_classes, **kwargs)


        






















