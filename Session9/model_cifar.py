from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt

import regularization

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dropout_value = 0.001, num_classes=10):
        super(Net, self).__init__()

        # Convolution Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(dropout_value)
        # Convolution Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Convolution Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Depthwise Separable Convolution Block
        self.depthwise_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.pointwise_conv = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Dilated Convolution Block
        self.dilated_conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn5 = nn.BatchNorm2d(128)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolution Block 1
        x = F.relu(self.dropout(self.bn1(self.conv1(x))))

        # Convolution Block 2
        x = F.relu(self.dropout(self.bn2(self.conv2(x))))

        # Convolution Block 3
        x = F.relu(self.dropout(self.bn3(self.conv3(x))))

        # Depthwise Separable Convolution Block
        x = F.relu(self.dropout(self.bn4(self.pointwise_conv(self.depthwise_conv(x)))))

        # Dilated Convolution Block
        x = F.relu(self.dropout(self.bn5(self.dilated_conv(x))))

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Fully Connected Layer
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)


def model():
    return Net()