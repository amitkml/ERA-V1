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

multiplier = 2

class GNet(nn.Module):
    def __init__(self, dropout_value=0.001):
        super(GNet, self).__init__()

        ## CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16 * multiplier, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(16 * multiplier, 16 * multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16 * multiplier, out_channels=32 * multiplier, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=32 * multiplier, out_channels=8 * multiplier, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(8 * multiplier, 8 * multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        ## CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8 * multiplier, out_channels=16 * multiplier, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.GroupNorm(16 * multiplier, 16 * multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16 * multiplier, out_channels=32 * multiplier, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=32 * multiplier, out_channels=8 * multiplier, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(8 * multiplier, 8 * multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        ## CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8 * multiplier, out_channels=16 * multiplier, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.GroupNorm(16 * multiplier, 16 * multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16 * multiplier, out_channels=32 * multiplier, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.Conv2d(in_channels=32 * multiplier, out_channels=8 * multiplier, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(8 * multiplier, 8 * multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )

        ## CONVOLUTION BLOCK 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8 * multiplier, out_channels=16 * multiplier, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.GroupNorm(16 * multiplier, 16 * multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )

        self.gap = nn.AvgPool2d(kernel_size=(6, 6))
        self.fc1 = nn.Linear(16 * multiplier, 10)

    def forward(self, x):
        x = self.pool1(self.convblock2(self.convblock1(x)))
        x = self.pool2(self.convblock4(self.convblock3(x)))
        x = self.convblock6(self.convblock5(x))
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=-1)
    
def model_gcifar():
    return GNet()