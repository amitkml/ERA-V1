## Assignment
Write a custom Links to an external site. ResNet architecture for CIFAR10 that has the following architecture:

- PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
- Layer1 -
  -  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
  -   R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
  - Add(X, R1)
- Layer 2 -
  - Conv 3x3 [256k]
  -  MaxPooling2D
  -    BN
  - ReLU
- Layer 3 -
  - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
  - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
  - Add(X, R2)
  - MaxPooling with Kernel Size 4
  - FC Layer 
- ​    SoftMax
- Uses One Cycle Policy such that:
  - Total Epochs = 24
  - Max at Epoch = 5
  - LRMIN = FIND
  - LRMAX = FIND
  - NO Annihilation
- Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
- Batch size = 512
- Use ADAM, and CrossEntropyLoss
- Target Accuracy: 90%

## Model Architecture

    
    class BasicBlock(nn.Module):
        """
        Basic block of the ResNet.
        """
    
        expansion = 1
    
        def __init__(self, in_planes, planes, norm_type, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.GroupNorm(1, planes) if norm_type == "LN" else nn.BatchNorm2d(planes)
    
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.GroupNorm(1, planes) if norm_type == "LN" else nn.BatchNorm2d(planes)
    
        def forward(self, x):
            """
            Forward method.
            """
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = F.relu(out)
            return out
    
    
    class CustomResNetClass(nn.Module):
        """
        ResNet Architecture.
        """
    
        def __init__(self, block, norm_type, num_classes=10):
            super().__init__()
    
            # Prep Layer
            self.prep = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),  # RF: 3x3
                nn.ReLU(),
                nn.BatchNorm2d(64),
            )
    
            # Conv Block
            self.conv_layer_1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),  # RF: 3x3
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
    
            # ResNet Block
            self.resblock_1 = block(128, 128, norm_type=norm_type)
    
            # Conv Block
            self.conv_layer_2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
    
            # Conv Block
            self.conv_layer_3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
    
            # ResNet Block
            self.resblock_2 = block(512, 512, norm_type=norm_type)
    
            # MaxPooling with Kernel Size 4
            self.pooling = nn.MaxPool2d(4, 4)
    
            # Fully Connected Layer
            self.fc1 = nn.Linear(512, num_classes)
    
        def forward(self, x):
            """
            Forward method.
            """
            out = self.prep(x)
            out = self.conv_layer_1(out)
    
            res_out_1 = self.resblock_1(out)
    
            # Residual Block
            out = out + res_out_1
            out = self.conv_layer_2(out)
            out = self.conv_layer_3(out)
    
            res_out_2 = self.resblock_2(out)
    
            # Residual Block
            out = out + res_out_2
            out = self.pooling(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            return out
    
    
    def CustomResNet(norm_type="BN"):
        """
        Custom ResNet model.
        """
        return CustomResNetClass(BasicBlock, norm_type)
    

## Model Training Logs



```
Epoch 1:
Train Loss=1.5479767322540283 Batch_id=390 LR= 0.01026 Train Accuracy= 43.60: 100%|██████████| 391/391 [00:50<00:00,  7.73it/s]

: Average Test loss: 0.0094, Test Accuracy: 6116/10000 (61.16%)

Epoch 2:
Train Loss=0.864130973815918 Batch_id=390 LR= 0.02103 Train Accuracy= 60.58: 100%|██████████| 391/391 [00:48<00:00,  8.13it/s]

: Average Test loss: 0.0067, Test Accuracy: 7088/10000 (70.88%)

Epoch 3:
Train Loss=0.7623922824859619 Batch_id=390 LR= 0.03434 Train Accuracy= 69.37: 100%|██████████| 391/391 [00:49<00:00,  7.85it/s]

: Average Test loss: 0.0057, Test Accuracy: 7430/10000 (74.30%)

Epoch 4:
Train Loss=0.7637758255004883 Batch_id=390 LR= 0.04509 Train Accuracy= 73.99: 100%|██████████| 391/391 [00:51<00:00,  7.64it/s]

: Average Test loss: 0.0068, Test Accuracy: 7122/10000 (71.22%)

Epoch 5:
Train Loss=0.7169293761253357 Batch_id=390 LR= 0.04919 Train Accuracy= 75.18: 100%|██████████| 391/391 [00:49<00:00,  7.91it/s]

: Average Test loss: 0.0055, Test Accuracy: 7534/10000 (75.34%)

Epoch 6:
Train Loss=0.6615993976593018 Batch_id=390 LR= 0.04885 Train Accuracy= 75.89: 100%|██████████| 391/391 [00:48<00:00,  8.11it/s]

: Average Test loss: 0.0057, Test Accuracy: 7631/10000 (76.31%)

Epoch 7:
Train Loss=0.640311598777771 Batch_id=390 LR= 0.04785 Train Accuracy= 76.75: 100%|██████████| 391/391 [00:51<00:00,  7.54it/s]

: Average Test loss: 0.0075, Test Accuracy: 6767/10000 (67.67%)

Epoch 8:
Train Loss=0.94514000415802 Batch_id=390 LR= 0.04622 Train Accuracy= 76.75: 100%|██████████| 391/391 [00:52<00:00,  7.41it/s]

: Average Test loss: 0.0067, Test Accuracy: 7234/10000 (72.34%)

Epoch 9:
Train Loss=0.784213662147522 Batch_id=390 LR= 0.04400 Train Accuracy= 77.52: 100%|██████████| 391/391 [00:49<00:00,  7.86it/s]

: Average Test loss: 0.0062, Test Accuracy: 7315/10000 (73.15%)

Epoch 10:
Train Loss=0.7646262645721436 Batch_id=390 LR= 0.04124 Train Accuracy= 77.91: 100%|██████████| 391/391 [00:52<00:00,  7.46it/s]

: Average Test loss: 0.0046, Test Accuracy: 8031/10000 (80.31%)

Epoch 11:
Train Loss=0.7545222043991089 Batch_id=390 LR= 0.03804 Train Accuracy= 78.29: 100%|██████████| 391/391 [00:47<00:00,  8.30it/s]

: Average Test loss: 0.0079, Test Accuracy: 6821/10000 (68.21%)

Epoch 12:
Train Loss=0.6461143493652344 Batch_id=390 LR= 0.03446 Train Accuracy= 78.89: 100%|██████████| 391/391 [00:49<00:00,  7.90it/s]

: Average Test loss: 0.0067, Test Accuracy: 7219/10000 (72.19%)

Epoch 13:
Train Loss=0.6778917908668518 Batch_id=390 LR= 0.03062 Train Accuracy= 79.33: 100%|██████████| 391/391 [00:50<00:00,  7.70it/s]

: Average Test loss: 0.0050, Test Accuracy: 7790/10000 (77.90%)

Epoch 14:
Train Loss=0.5851505994796753 Batch_id=390 LR= 0.02662 Train Accuracy= 79.93: 100%|██████████| 391/391 [00:52<00:00,  7.44it/s]

: Average Test loss: 0.0051, Test Accuracy: 7812/10000 (78.12%)

Epoch 15:
Train Loss=0.6611266136169434 Batch_id=390 LR= 0.02255 Train Accuracy= 80.62: 100%|██████████| 391/391 [00:50<00:00,  7.81it/s]

: Average Test loss: 0.0056, Test Accuracy: 7580/10000 (75.80%)

Epoch 16:
Train Loss=0.6180647015571594 Batch_id=390 LR= 0.01855 Train Accuracy= 81.66: 100%|██████████| 391/391 [00:48<00:00,  7.99it/s]

: Average Test loss: 0.0053, Test Accuracy: 7681/10000 (76.81%)

Epoch 17:
Train Loss=0.6460245251655579 Batch_id=390 LR= 0.01471 Train Accuracy= 82.36: 100%|██████████| 391/391 [00:46<00:00,  8.44it/s]

: Average Test loss: 0.0041, Test Accuracy: 8221/10000 (82.21%)

Epoch 18:
Train Loss=0.6025124192237854 Batch_id=390 LR= 0.01113 Train Accuracy= 83.91: 100%|██████████| 391/391 [00:46<00:00,  8.35it/s]

: Average Test loss: 0.0034, Test Accuracy: 8505/10000 (85.05%)

Epoch 19:
Train Loss=0.48781028389930725 Batch_id=390 LR= 0.00793 Train Accuracy= 85.03: 100%|██████████| 391/391 [00:48<00:00,  8.04it/s]

: Average Test loss: 0.0033, Test Accuracy: 8620/10000 (86.20%)

Epoch 20:
Train Loss=0.38796767592430115 Batch_id=390 LR= 0.00518 Train Accuracy= 87.39: 100%|██████████| 391/391 [00:46<00:00,  8.33it/s]

: Average Test loss: 0.0029, Test Accuracy: 8797/10000 (87.97%)

Epoch 21:
Train Loss=0.22787952423095703 Batch_id=390 LR= 0.00296 Train Accuracy= 89.59: 100%|██████████| 391/391 [00:51<00:00,  7.64it/s]

: Average Test loss: 0.0023, Test Accuracy: 9001/10000 (90.01%)

Epoch 22:
Train Loss=0.1879851073026657 Batch_id=390 LR= 0.00133 Train Accuracy= 92.51: 100%|██████████| 391/391 [00:46<00:00,  8.48it/s]

: Average Test loss: 0.0021, Test Accuracy: 9111/10000 (91.11%)

Epoch 23:
Train Loss=0.14682181179523468 Batch_id=390 LR= 0.00033 Train Accuracy= 94.65: 100%|██████████| 391/391 [00:49<00:00,  7.82it/s]

: Average Test loss: 0.0018, Test Accuracy: 9194/10000 (91.94%)

Epoch 24:
Train Loss=0.06818695366382599 Batch_id=390 LR= 0.00000 Train Accuracy= 96.02: 100%|██████████| 391/391 [00:47<00:00,  8.16it/s]

: Average Test loss: 0.0018, Test Accuracy: 9225/10000 (92.25%)
```

## Model class wise accuracy

```
Accuracy of airplane : 94 %
Accuracy of automobile : 96 %
Accuracy of  bird : 86 %
Accuracy of   cat : 82 %
Accuracy of  deer : 94 %
Accuracy of   dog : 87 %
Accuracy of  frog : 94 %
Accuracy of horse : 95 %
Accuracy of  ship : 95 %
Accuracy of truck : 95 %
```

## Misclassified Images

![im](https://github.com/amitkml/ERA-V1/blob/main/Session10/miss_classified_images.PNG?raw=true)

## LR Finder output

![im](https://github.com/amitkml/ERA-V1/blob/main/Session10/lr_finder.PNG?raw=true)

## Loss details during Model training and evaluation

![im](https://github.com/amitkml/ERA-V1/blob/main/Session10/test_evaluation_result.PNG?raw=true)

