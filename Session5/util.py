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

def prepare_data_loader():
  
  '''
    Parameters
    - This function does not accept any parameters.

    Returns
    - train_loader (torch.utils.data.DataLoader): A data loader object containing the training set of the FashionMNIST dataset.
    
    Steps
    - Create a FashionMNIST dataset object called train_set using the torchvision.datasets.FashionMNIST class. The dataset is configured with the following parameters:
        root='./data/FashionMNIST': Specifies the directory to store the dataset.
        train=True: Indicates that the dataset is for training purposes.
        download=True: Specifies to download the dataset if it is not already present.
        transform=transforms.Compose([transforms.ToTensor()]): Applies a series of transformations to the dataset, converting the images to tensors.
    - Create a data loader object called train_loader using the torch.utils.data.DataLoader class. The data loader is configured with the following parameters:
        train_set: The training set of the FashionMNIST dataset.
        batch_size=100: Specifies the number of samples per batch.
        shuffle=True: Randomly shuffles the samples during training.
    - Return the train_loader object.  
  '''
  train_set = torchvision.datasets.FashionMNIST(
        root='./fmistdata'
        ,train=True
        ,download=True
        ,transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
  train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size = 32, 
        shuffle = True
    )
  return train_loader

def show_images_batch_from_loader(loader):
  '''
    This function is used to display a batch of images along with their corresponding labels from a data loader.

    Parameters:
    - loader (torch.utils.data.DataLoader): A data loader object that contains the images and labels.
    
    Returns:
    - None
    
    Steps:
    - Retrieve the next batch of images and labels from the data loader using next(iter(loader)).
    - Assign the images and labels to the variables images and labels, respectively.
    - Create a grid of images using torchvision.utils.make_grid(images, nrow=10). The nrow parameter specifies the number of images to display in each row of the grid.
    - Display the grid of images using plt.imshow(np.transpose(grid, (1, 2, 0))). The grid is transposed to match the expected image dimensions.
    - Print the corresponding labels using print('labels:', labels
  '''  
  batch = next(iter(loader))
  images, labels = batch
  grid = torchvision.utils.make_grid(images, nrow=10)
  plt.figure(figsize=(15,15))
  plt.imshow(np.transpose(grid, (1,2,0)))
  print('labels:', labels)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
  
  
def train_model(model, optimizer, epochs):
    train_loader = prepare_data_loader()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        for batch in train_loader:
            images, labels = batch
            preds = model(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
        print("epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)
    