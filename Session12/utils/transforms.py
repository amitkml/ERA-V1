import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torchmetrics import Accuracy
from utils.helper import seed_everything, get_default_device, calculate_mean_std



def apply_transforms_custom_resnet(mean, std):
    """
    Image augmentations for train and test set.
    """
    train_transforms = A.Compose(
        [
            # RandomCrop with Padding
            A.Sequential(
                [
                    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                    A.RandomCrop(width=32, height=32, p=1),
                ],
                p=1,
            ),
            # Horizontal Flipping
            A.HorizontalFlip(p=1),
            # Cutout
            A.CoarseDropout(
                max_holes=3,
                max_height=8,
                max_width=8,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=tuple((x * 255.0 for x in mean)),
                p=0.8,
            ),
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    return (
        lambda img: train_transforms(image=np.array(img))["image"],
        lambda img: test_transforms(image=np.array(img))["image"],
    )

# # Load CIFAR-10 dataset and create dataloaders
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=apply_transforms_custom_resnet(mean, std))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=apply_transforms_custom_resnet(mean, std))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)


# Define LightningDataModule for data loading
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=128):
        
        super().__init__()
        self.mean, self.std = calculate_mean_std("CIFAR10")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transforms, self.test_transforms = apply_transforms_custom_resnet(mean, std)
        self.train_dataset = train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.train_transforms)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.test_transforms)
        self.transform = apply_transforms_custom_resnet(self.mean,self.std)

    def prepare_data(self):
        # Download the dataset if needed
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, transform=self.train_transforms)
        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
    

def apply_transforms(mean, std):
    """
    Image augmentations for train and test set.
    """
    train_transforms = A.Compose(
        [
            A.Sequential(
                [
                    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                    A.RandomCrop(width=32, height=32, p=1),
                ],
                p=0.5,
            ),
            A.Rotate(limit=5, p=0.2),
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=tuple((x * 255.0 for x in mean)),
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    return (
        lambda img: train_transforms(image=np.array(img))["image"],
        lambda img: test_transforms(image=np.array(img))["image"],
    )


# # def apply_transforms_custom_resnet(mean, std):
# #     """
# #     Image augmentations for train and test set.
# #     """
# #     train_transforms = A.Compose(
# #         [
# #             # RandomCrop with Padding
# #             A.Sequential(
# #                 [
# #                     A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
# #                     A.RandomCrop(width=32, height=32, p=1),
# #                 ],
# #                 p=1,
# #             ),
# #             # Horizontal Flipping
# #             A.HorizontalFlip(p=1),
# #             # Cutout
# #             A.CoarseDropout(
# #                 max_holes=3,
# #                 max_height=8,
# #                 max_width=8,
# #                 min_holes=1,
# #                 min_height=8,
# #                 min_width=8,
# #                 fill_value=tuple((x * 255.0 for x in mean)),
# #                 p=0.8,
# #             ),
# #             A.Normalize(mean=mean, std=std, always_apply=True),
# #             ToTensorV2(),
# #         ]
# #     )

# #     test_transforms = A.Compose(
# #         [
# #             A.Normalize(mean=mean, std=std, always_apply=True),
# #             ToTensorV2(),
# #         ]
# #     )

# #     return (
# #         lambda img: train_transforms(image=np.array(img))["image"],
# #         lambda img: test_transforms(image=np.array(img))["image"],
# #     )


# def apply_transforms_tiny_imagenet(mean, std):
#     """
#     Image augmentations for train and test set for Tiny ImageNet.
#     """
#     train_transforms = A.Compose(
#         [
#             # RandomCrop with Padding
#             A.Sequential(
#                 [
#                     A.PadIfNeeded(min_height=72, min_width=72, always_apply=True),
#                     A.RandomCrop(width=64, height=64, p=1),
#                 ],
#                 p=1,
#             ),
#             # Horizontal Flipping
#             A.HorizontalFlip(p=0.5),
#             # Rotate +- 5 degrees
#             A.Rotate(limit=5),
#             # Cutout
#             A.CoarseDropout(
#                 max_holes=2,
#                 max_height=32,
#                 max_width=32,
#                 min_holes=1,
#                 min_height=32,
#                 min_width=32,
#                 fill_value=tuple((x * 255.0 for x in mean)),
#                 p=0.8,
#             ),
#             A.Normalize(mean=mean, std=std, always_apply=True),
#             ToTensorV2(),
#         ]
#     )

#     test_transforms = A.Compose(
#         [
#             A.Normalize(mean=mean, std=std, always_apply=True),
#             ToTensorV2(),
#         ]
#     )

#     return (
#         lambda img: train_transforms(image=np.array(img))["image"],
#         lambda img: test_transforms(image=np.array(img))["image"],
#     )
