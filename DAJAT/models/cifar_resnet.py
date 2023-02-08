
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AdversarialDefensiveModule
from .layerops import Sequential


class BasicBlock(AdversarialDefensiveModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn1_auto = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_auto = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = Sequential()
        self.shortcut_bn = Sequential()
        self.shortcut_bn_auto = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
            self.shortcut_bn = Sequential(
                nn.BatchNorm2d(self.expansion * planes)
            )

            self.shortcut_bn_auto = Sequential(
                nn.BatchNorm2d(self.expansion * planes)
            )


    def forward(self, x):
        #print(input)
        out = x[0]
        batch_norm = x[1]
        
        out1 = self.conv1(out)
        if batch_norm=='base':
            out1 = self.relu(self.bn1(out1))
        elif batch_norm=='auto':
            out1 = self.relu(self.bn1_auto(out1))
        out1 = self.conv2(out1)     
        if batch_norm=='base':
            out1 = self.bn2(out1)
        elif batch_norm=='auto':
            out1 = self.bn2_auto(out1)

        feat = self.shortcut(out)
        if batch_norm=='base':
            feat2 = self.shortcut_bn(feat)
        elif batch_norm=='auto':
            feat2 = self.shortcut_bn_auto(feat)
        out1 += feat2
        out1 = F.relu(out1)
        return [out1,batch_norm]


class Bottleneck(AdversarialDefensiveModule):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, batch_norm):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(AdversarialDefensiveModule):
    def __init__(self, block, num_blocks, num_classes=10, strides='1222'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        strides = list(map(int, strides))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_auto = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=strides[3])
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def forward(self, x, batch_norm='base'):
        out = self.conv1(x)
        if batch_norm=='base':
            out = F.relu(self.bn1(out))
        elif batch_norm=='auto':
            out = F.relu(self.bn1_auto(out))

        out = self.layer1([out, batch_norm])
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out[0], 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes, strides):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, strides=strides)