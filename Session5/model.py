import torch
import numpy as np
import torch.nn as nn
import torch
import torchvision # provide access to datasets, models, transforms, utils, etc
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

torch.set_printoptions(linewidth=120)


class Network(nn.Module):
  def __init__(self):
    super().__init__()
    # input 28 # output 24 # receptive_field = 5
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) 
    # input 24 # output 20 # receptive_field = 9
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    # input 12x20x20, output 120
    # input 10*512
    self.fc1 = nn.Linear(in_features=192, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=10)
  
  def forward(self, t):
    # input layer
    x = t

    # conv1 layer
    x = self.conv1(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2) # 28 | 24 | 12

    # conv2 layer
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2) # 12 | 8 | 4 >> 12x4x4
    # print(x.shape)
    # reshapre
    # x = x.reshape(-1, 12 * 4 * 4)
    x = x.view(x.size(0),-1)
    # print(x.shape)
    # fc1 layer
    x = self.fc1(x)
    x = F.relu(x)

    # fc2 layer
    x = self.fc2(x)
    x = F.relu(x)

    # output layer
    x = self.out(x)
    # x = F.softmax(x, dim=1)
    return x
  
  def show_parameters_layer(self):
    for name, param in self.named_parameters():
      print(name, "\t\t", param.shape)
