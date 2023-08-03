
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy


class BasicBlock(nn.Module):
    """
    Basic block of the ResNet.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

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


class CustomResNetClass(pl.LightningModule):
    """
    ResNet Architecture.
    """

    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10, max_lr=0.1, 
                 steps_per_epoch=None, div_factor=10, pct_start=0.3,
                 lambda_l1=0.0, grad_clip=None):
        super(CustomResNetClass, self).__init__()
        self.in_planes = 64

        self.lambda_l1 = lambda_l1
        self.grad_clip = grad_clip

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.train_acc = Accuracy(num_classes=num_classes)

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
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)

        loss = F.cross_entropy(y_pred, target)

        # L1 Regularization
        if self.lambda_l1 > 0:
            l1 = 0
            for p in self.parameters():
                l1 = l1 + p.abs().sum()
                loss = loss + self.lambda_l1 * l1

        self.log("train_loss", loss, prog_bar=True)
        acc = self.train_acc(y_pred, target)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = F.cross_entropy(y_pred, target)
        self.log("val_loss", loss, prog_bar=True)
        acc = self.train_acc(y_pred, target)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
         # Create OneCycleLR scheduler
        scheduler = OneCycleLR(optimizer, max_lr=self.max_lr, 
                               steps_per_epoch=self.steps_per_epoch,
                                 div_factor=self.div_factor, 
                                 pct_start=self.pct_start)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def test_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = F.cross_entropy(y_pred, target)
        self.log("test_loss", loss, prog_bar=True)
        acc = self.test_acc(y_pred, target)
        self.log("test_acc", acc, prog_bar=True)

# import torch.nn as nn
# import torch.nn.functional as F


# class BasicBlock(nn.Module):
#     """
#     Basic block of the ResNet.
#     """

#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class CustomResNetClass(nn.Module):
#     """
#     ResNet Architecture.
#     """

#     def __init__(self, block, num_blocks, num_classes=10):
#         super(CustomResNetClass, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


def CustomResNet(norm_type="BN"):
    """
    Custom ResNet model.
    """
    return CustomResNetClass(BasicBlock, [2, 2, 2, 2])
