## Assignment
- Assignment: 

  1. Check this Repo out: https://github.com/kuangliu/pytorch-cifar
  2. (Optional) You are going to follow the same structure for your Code (as a reference). So Create:
     1. models folder - this is  where you'll add all of your future models. Copy resnet.py into this  folder, this file should only have ResNet 18/34 models. **Delete Bottleneck Class**
     2. main.py - from Google Colab, now onwards, this is the file that you'll import (along with the  model). Your main file shall be able to take these params or you should  be able to pull functions from it and then perform operations, like  (including but not limited to):
        1. training and test loops
        2. data split between test and train
        3. epochs
        4. batch size
        5. which optimizer to run
        6. do we run a scheduler?
     3. utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:
        1. image transforms,
        2. gradcam,
        3. misclassification code,
        4. tensorboard related stuff
        5. advanced training policies, etc
        6. etc
  3. Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
     1. pull your Github code to google colab (don't copy-paste code)
     2. prove that you are following the above structure
     3. that the code in your google collab notebook is NOTHING.. barely  anything. There should not be any function or class that you can define  in your Google Colab Notebook. Everything must be imported from all of  your other files
     4. your colab file must:
        1. train resnet18 for 20 epochs on the CIFAR10 dataset
        2. show loss curves for test and train datasets
        3. show a gallery of 10 misclassified images
        4. show [gradcam     ](https://github.com/jacobgil/pytorch-grad-cam)

  1. [  Links to an external site.](https://github.com/jacobgil/pytorch-grad-cam) output on 10 misclassified images. **Remember if you are applying GradCAM on a channel that is less than 5px, then  please don't bother to submit the assignment. 😡🤬🤬🤬🤬**

  Once done, upload the code to GitHub, and share the code. This  readme must link to the main repo so we can read your file structure. 

  Train for 20 epochs

  Get 10 misclassified images

  Get 10 GradCam outputs on any **misclassified images (remember that you MUST use the library we discussed in the class)**

  Apply these transforms while training:

  1. RandomCrop(32, padding=4)
  2. CutOut(16x16)

## Model Architecture


    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 64, 32, 32]           1,728
           BatchNorm2d-2           [-1, 64, 32, 32]             128
                Conv2d-3           [-1, 64, 32, 32]          36,864
           BatchNorm2d-4           [-1, 64, 32, 32]             128
                Conv2d-5           [-1, 64, 32, 32]          36,864
           BatchNorm2d-6           [-1, 64, 32, 32]             128
            BasicBlock-7           [-1, 64, 32, 32]               0
                Conv2d-8           [-1, 64, 32, 32]          36,864
           BatchNorm2d-9           [-1, 64, 32, 32]             128
               Conv2d-10           [-1, 64, 32, 32]          36,864
          BatchNorm2d-11           [-1, 64, 32, 32]             128
           BasicBlock-12           [-1, 64, 32, 32]               0
               Conv2d-13          [-1, 128, 16, 16]          73,728
          BatchNorm2d-14          [-1, 128, 16, 16]             256
               Conv2d-15          [-1, 128, 16, 16]         147,456
          BatchNorm2d-16          [-1, 128, 16, 16]             256
               Conv2d-17          [-1, 128, 16, 16]           8,192
          BatchNorm2d-18          [-1, 128, 16, 16]             256
           BasicBlock-19          [-1, 128, 16, 16]               0
               Conv2d-20          [-1, 128, 16, 16]         147,456
          BatchNorm2d-21          [-1, 128, 16, 16]             256
               Conv2d-22          [-1, 128, 16, 16]         147,456
          BatchNorm2d-23          [-1, 128, 16, 16]             256
           BasicBlock-24          [-1, 128, 16, 16]               0
               Conv2d-25            [-1, 256, 8, 8]         294,912
          BatchNorm2d-26            [-1, 256, 8, 8]             512
               Conv2d-27            [-1, 256, 8, 8]         589,824
          BatchNorm2d-28            [-1, 256, 8, 8]             512
               Conv2d-29            [-1, 256, 8, 8]          32,768
          BatchNorm2d-30            [-1, 256, 8, 8]             512
           BasicBlock-31            [-1, 256, 8, 8]               0
               Conv2d-32            [-1, 256, 8, 8]         589,824
          BatchNorm2d-33            [-1, 256, 8, 8]             512
               Conv2d-34            [-1, 256, 8, 8]         589,824
          BatchNorm2d-35            [-1, 256, 8, 8]             512
           BasicBlock-36            [-1, 256, 8, 8]               0
               Conv2d-37            [-1, 512, 4, 4]       1,179,648
          BatchNorm2d-38            [-1, 512, 4, 4]           1,024
               Conv2d-39            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-40            [-1, 512, 4, 4]           1,024
               Conv2d-41            [-1, 512, 4, 4]         131,072
          BatchNorm2d-42            [-1, 512, 4, 4]           1,024
           BasicBlock-43            [-1, 512, 4, 4]               0
               Conv2d-44            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-45            [-1, 512, 4, 4]           1,024
               Conv2d-46            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-47            [-1, 512, 4, 4]           1,024
           BasicBlock-48            [-1, 512, 4, 4]               0
               Linear-49                   [-1, 10]           5,130
    ================================================================
    Total params: 11,173,962
    Trainable params: 11,173,962
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 11.25
    Params size (MB): 42.63
    Estimated Total Size (MB): 53.89


​    


​    

## Model Training Logs



```
Epoch 1:
Train Loss=1.4405291080474854 Batch_id=390 LR= 0.07723 Train Accuracy= 33.89: 100%|██████████| 391/391 [01:59<00:00,  3.26it/s]

: Average Test loss: 0.0133, Test Accuracy: 3858/10000 (38.58%)

Epoch 2:
Train Loss=1.3430726528167725 Batch_id=390 LR= 0.15828 Train Accuracy= 43.32: 100%|██████████| 391/391 [01:49<00:00,  3.58it/s]

: Average Test loss: 0.0353, Test Accuracy: 1983/10000 (19.83%)

Epoch 3:
Train Loss=1.5983365774154663 Batch_id=390 LR= 0.25842 Train Accuracy= 44.12: 100%|██████████| 391/391 [01:55<00:00,  3.38it/s]

: Average Test loss: 0.0240, Test Accuracy: 1899/10000 (18.99%)

Epoch 4:
Train Loss=1.6557127237319946 Batch_id=390 LR= 0.33937 Train Accuracy= 42.68: 100%|██████████| 391/391 [01:52<00:00,  3.48it/s]

: Average Test loss: 0.0164, Test Accuracy: 2865/10000 (28.65%)

Epoch 5:
Train Loss=1.5967676639556885 Batch_id=390 LR= 0.37018 Train Accuracy= 41.23: 100%|██████████| 391/391 [01:46<00:00,  3.67it/s]

: Average Test loss: 0.0164, Test Accuracy: 2986/10000 (29.86%)

Epoch 6:
Train Loss=1.7119026184082031 Batch_id=390 LR= 0.36764 Train Accuracy= 41.46: 100%|██████████| 391/391 [01:55<00:00,  3.38it/s]

: Average Test loss: 0.0156, Test Accuracy: 3337/10000 (33.37%)

Epoch 7:
Train Loss=1.5947458744049072 Batch_id=390 LR= 0.36013 Train Accuracy= 41.48: 100%|██████████| 391/391 [01:44<00:00,  3.75it/s]

: Average Test loss: 0.0241, Test Accuracy: 1562/10000 (15.62%)

Epoch 8:
Train Loss=1.693007230758667 Batch_id=390 LR= 0.34784 Train Accuracy= 41.18: 100%|██████████| 391/391 [01:52<00:00,  3.46it/s]

: Average Test loss: 0.0149, Test Accuracy: 3661/10000 (36.61%)

Epoch 9:
Train Loss=1.351563811302185 Batch_id=390 LR= 0.33111 Train Accuracy= 42.07: 100%|██████████| 391/391 [01:51<00:00,  3.51it/s]

: Average Test loss: 0.0380, Test Accuracy: 2068/10000 (20.68%)

Epoch 10:
Train Loss=1.6115176677703857 Batch_id=390 LR= 0.31039 Train Accuracy= 41.96: 100%|██████████| 391/391 [01:57<00:00,  3.32it/s]

: Average Test loss: 0.0160, Test Accuracy: 3063/10000 (30.63%)

Epoch 11:
Train Loss=1.8349758386611938 Batch_id=390 LR= 0.28626 Train Accuracy= 42.06: 100%|██████████| 391/391 [01:39<00:00,  3.95it/s]

: Average Test loss: 0.0151, Test Accuracy: 3245/10000 (32.45%)

Epoch 12:
Train Loss=1.541418433189392 Batch_id=390 LR= 0.25937 Train Accuracy= 42.33: 100%|██████████| 391/391 [01:53<00:00,  3.46it/s]

: Average Test loss: 0.0170, Test Accuracy: 3250/10000 (32.50%)

Epoch 13:
Train Loss=1.4480793476104736 Batch_id=390 LR= 0.23045 Train Accuracy= 42.92: 100%|██████████| 391/391 [01:47<00:00,  3.65it/s]

: Average Test loss: 0.0337, Test Accuracy: 2190/10000 (21.90%)

Epoch 14:
Train Loss=1.5729821920394897 Batch_id=390 LR= 0.20030 Train Accuracy= 43.26: 100%|██████████| 391/391 [01:51<00:00,  3.49it/s]

: Average Test loss: 0.0335, Test Accuracy: 1543/10000 (15.43%)

Epoch 15:
Train Loss=1.5440442562103271 Batch_id=390 LR= 0.16973 Train Accuracy= 44.59: 100%|██████████| 391/391 [01:44<00:00,  3.75it/s]

: Average Test loss: 0.0238, Test Accuracy: 2134/10000 (21.34%)

Epoch 16:
Train Loss=1.2687945365905762 Batch_id=390 LR= 0.13958 Train Accuracy= 45.62: 100%|██████████| 391/391 [01:58<00:00,  3.31it/s]

: Average Test loss: 0.0322, Test Accuracy: 2330/10000 (23.30%)

Epoch 17:
Train Loss=1.4898033142089844 Batch_id=390 LR= 0.11067 Train Accuracy= 46.85: 100%|██████████| 391/391 [01:45<00:00,  3.70it/s]

: Average Test loss: 0.0141, Test Accuracy: 3899/10000 (38.99%)

Epoch 18:
Train Loss=1.227722406387329 Batch_id=390 LR= 0.08379 Train Accuracy= 49.20: 100%|██████████| 391/391 [01:41<00:00,  3.84it/s]

: Average Test loss: 0.0113, Test Accuracy: 4885/10000 (48.85%)

Epoch 19:
Train Loss=1.3949377536773682 Batch_id=390 LR= 0.05968 Train Accuracy= 50.78: 100%|██████████| 391/391 [01:44<00:00,  3.73it/s]

: Average Test loss: 0.0146, Test Accuracy: 4397/10000 (43.97%)

Epoch 20:
Train Loss=1.423964023590088 Batch_id=390 LR= 0.03898 Train Accuracy= 53.30: 100%|██████████| 391/391 [01:47<00:00,  3.64it/s]

: Average Test loss: 0.0147, Test Accuracy: 4239/10000 (42.39%)

Epoch 21:
Train Loss=0.966474711894989 Batch_id=390 LR= 0.02228 Train Accuracy= 56.68: 100%|██████████| 391/391 [01:41<00:00,  3.84it/s]

: Average Test loss: 0.0118, Test Accuracy: 4625/10000 (46.25%)

Epoch 22:
Train Loss=0.9926832318305969 Batch_id=390 LR= 0.01001 Train Accuracy= 61.63: 100%|██████████| 391/391 [01:47<00:00,  3.65it/s]

: Average Test loss: 0.0083, Test Accuracy: 6289/10000 (62.89%)

Epoch 23:
Train Loss=0.8653895258903503 Batch_id=390 LR= 0.00252 Train Accuracy= 67.61: 100%|██████████| 391/391 [01:44<00:00,  3.73it/s]

: Average Test loss: 0.0059, Test Accuracy: 7332/10000 (73.32%)

Epoch 24:
Train Loss=0.7106490135192871 Batch_id=390 LR= 0.00000 Train Accuracy= 72.86: 100%|██████████| 391/391 [01:43<00:00,  3.77it/s]

: Average Test loss: 0.0051, Test Accuracy: 7759/10000 (77.59%)
```

## Model class wise accuracy

```
Accuracy of airplane : 80 %
Accuracy of automobile : 92 %
Accuracy of  bird : 61 %
Accuracy of   cat : 59 %
Accuracy of  deer : 70 %
Accuracy of   dog : 67 %
Accuracy of  frog : 86 %
Accuracy of horse : 80 %
Accuracy of  ship : 88 %
Accuracy of truck : 87 %
```

## Misclassified Images

![im](https://github.com/amitkml/ERA-V1/blob/main/Session11/miss_classified_images.PNG?raw=true)

## LR Finder output

![im](https://github.com/amitkml/ERA-V1/blob/main/Session10/lr_finder.PNG?raw=true)

## Loss details during Model training and evaluation

![im](https://github.com/amitkml/ERA-V1/blob/main/Session11/test_evaluation_result.PNG?raw=true)

## Gradcam

![im](https://github.com/amitkml/ERA-V1/blob/main/Session11/grad_Cam.PNG?raw=true)
