"""
model.py

ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_planes, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_planes[0]

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, num_planes[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_planes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_planes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_planes[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(num_planes[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.shape) # 4, 512
        out = self.linear(out)
        return out


def MyResNet(block=BasicBlock, num_planes=[64, 128, 256, 512], num_blocks=[2, 1, 1, 1]):
    return ResNet(block, num_planes, num_blocks)


def ResNet18():
    return ResNet(BasicBlock, [64, 128, 256, 512], [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [64, 128, 256, 512], [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [64, 128, 256, 512], [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [64, 128, 256, 512], [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [64, 128, 256, 512], [3, 8, 36, 3])

if __name__ == '__main__':
    # net = ResNet18()
    # net = MyResNet(BasicBlock, [32, 64, 128, 256], [2, 2, 2, 2])
    net = MyResNet(BasicBlock, [64, 128, 256, 512], [2, 1, 1, 1])

    torchsummary.summary(net, (3, 32, 32))
    inp = torch.randn(4, 3, 32, 32)
    res = net(inp)
    print(res.shape)
    # resnet18: 11,173,962
    # resnet34: 21,282,122
    # MyResNet: 4,977,226