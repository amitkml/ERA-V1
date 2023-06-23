# Fundamentals 

## Batch Normalization

Batch normalization (also known as batch norm) is **a method used to make training of artificial neural networks faster and  more stable through normalization of the layers' inputs by re-centering  and re-scaling**.

For each batch in the input dataset, the mini-batch gradient descent  algorithm runs its updates. It updates the weights and biases  (parameters) of the neural network so as to fit to the distribution seen at the input to the specific layer for the current batch.

As an example, let’s consider a mini-batch with 3 input samples, each input vector being four features long. Here’s a simple illustration of  how the mean and standard deviation are computed in this case. Once we  compute the mean and standard deviation, we can subtract the mean and  divide by the standard deviation.

![Batch Normalization Example](https://d33wubrfki0l68.cloudfront.net/5863322b42dcdf4b45ffef4de43f6ef0385db477/e6251/images/batch-normalization-example.png)

- forcing all the pre-activations to be zero and unit standard deviation  across all batches can be too restrictive. It may be the case that the  fluctuant distributions are necessary for the network to learn certain  classes better.
- To address this, batch normalization introduces two parameters: a scaling factor `gamma` (γ) and an offset `beta` (β). These are learnable parameters, so if the fluctuation in input  distribution is necessary for the neural network to learn a certain  class better, then the network learns the optimal values of `gamma` and `beta` for each mini-batch. T

Two limitations of batch normalization can arise:

- In batch normalization, we use the *batch statistics*: the mean and standard deviation corresponding to the current  mini-batch. However, when the batch size is small, the sample mean and  sample standard deviation are not representative enough of the actual  distribution and the network cannot learn anything meaningful.
- As batch normalization depends on batch statistics for normalization, it  is less suited for sequence models. This is because, in sequence models, we may have sequences of potentially different lengths and smaller  batch sizes corresponding to longer sequences.

## Layer Normalization

In layer normalization, all neurons in a particular layer effectively have the same distribution across all features for a given input.

For example, if each input has `d` features, it’s a d-dimensional vector. If there are `B` elements in a batch, the normalization is done along the length of the d-dimensional vector and not across the batch of size `B`.

Normalizing *across all features* but for each of the inputs to a specific layer removes the dependence  on batches. This makes layer normalization well suited for sequence  models such as [transformers](https://www.pinecone.io/learn/sentence-embeddings/) and [recurrent neural networks (RNNs)](https://www.ibm.com/cloud/learn/recurrent-neural-networks) that were popular in the pre-transformer era.

We consider the example of a mini-batch containing three input samples, each with four features.

![Layer Normalization](https://d33wubrfki0l68.cloudfront.net/c8f1f7a886548f82234f8a3b06faeecfbb88c657/42d49/images/layer-normalization.png)

## Group Normalization

Finally, for group norm, the batch is first divided into groups (32 by default, discussed later). The batch with dimension `(N, C, W, H)` is first reshaped to `(N, G, C//G, H, W)` dimensions where `G` represents the **number of groups**. Finally, the *mean* and *std deviation* are calculated along the groups, that is `(H, W)` and along `C//G` channels. This is also illustrated very well in `fig-4`.

One key thing to note here, if `C == G`, that is the number of groups are set to be equal to the number of channels (one channel per group), then **GN** becomes **IN**.

And if, `G == 1`, that is number of groups is set to 1, **GN** becomes **LN**

![im](https://amaarora.github.io/images/gn_explained.jpg)

# Assignment and Design

## Assignment

Change the dataset to CIFAR10

Make this network:

1. C1 C2 ***c3 P1*** C3 C4 C5 ***c6 P2*** C7 C8 C9 GAP C10
2. Keep the parameter count less than 50000
3. Try and add one layer to another
4. Max Epochs is 20

You are making 3 versions of the above code (in each case achieve above 70% accuracy):

1. Network with Group Normalization
2. Network with Layer Normalization
3. Network with Batch Normalization

Share these details

1. 1. Training accuracy for 3 models
   2. Test accuracy for 3 models

## BN Network

The network has been designed as sequential one.

### Design

The provided network is designed for image classification on the CIFAR-10 dataset. It consists of multiple convolutional blocks followed by pooling and fully connected layers. Here's a breakdown of the network architecture:

1. **Convolution Block 1:**
   - Input: 3-channel images
   - Convolutional layer with 16 output channels, 3x3 kernel, stride 1, and padding 1
   - Batch normalization
   - Dropout with a specified dropout value
   - ReLU activation function
2. **Convolution Block 2:**
   - Convolutional layer with 16 input channels and 32 output channels, 3x3 kernel, and padding 1
   - Grouped convolution with 16 groups
   - Convolutional layer with 32 input channels and 8 output channels, 1x1 kernel
   - Batch normalization
   - Dropout with a specified dropout value
   - ReLU activation function
3. **Pooling Layer 1:**
   - Max pooling with a 2x2 kernel and stride 2
4. **Convolution Block 3:**
   - Convolutional layer with 8 input channels and 16 output channels, 3x3 kernel, padding 1, and dilation 1
   - Batch normalization
   - Dropout with a specified dropout value
   - ReLU activation function
5. **Convolution Block 4:**
   - Convolutional layer with 16 input channels and 32 output channels, 3x3 kernel, and padding 1
   - Grouped convolution with 16 groups
   - Convolutional layer with 32 input channels and 8 output channels, 1x1 kernel
   - Batch normalization
   - Dropout with a specified dropout value
   - ReLU activation function
6. **Pooling Layer 2:**
   - Max pooling with a 2x2 kernel and stride 2
7. **Convolution Block 5:**
   - Convolutional layer with 8 input channels and 16 output channels, 3x3 kernel, padding 1, and dilation 2
   - Batch normalization
   - Dropout with a specified dropout value
   - ReLU activation function
8. **Convolution Block 6:**
   - Convolutional layer with 16 input channels and 32 output channels, 3x3 kernel, padding 1, and dilation 1
   - Grouped convolution with 16 groups
   - Convolutional layer with 32 input channels and 8 output channels, 1x1 kernel
   - Batch normalization
   - Dropout with a specified dropout value
   - ReLU activation function
9. **Convolution Block 7:**
   - Convolutional layer with 8 input channels and 16 output channels, 3x3 kernel, padding 1, and dilation 1
   - Batch normalization
   - Dropout with a specified dropout value
   - ReLU activation function
10. **Global Average Pooling Layer:**
    - Average pooling layer with a kernel size of 6x6
11. **Fully Connected Layer:**
    - Linear layer with input size 16 and output size 10 (number of classes in CIFAR-10)
12. **Output Layer:**
    - Log-softmax activation function to obtain the class probabilities

The network is trained using the cross-entropy loss function and optimized using the stochastic gradient descent (SGD) optimizer. The learning rate, weight decay, and momentum can be adjusted based on the training requirements.

```
class Net(nn.Module):
    def __init__(self,dropout_value = 0):
        super(Net, self).__init__()
    
        
        ## CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3,3), stride=1, padding=1),
#             nn.Conv2d(in_channels=3, out_channels=33, kernel_size=(3, 3), padding=1, groups = 3, bias=False),
#             nn.Conv2d(in_channels=33, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 32 output_size = 32 receptive_field = 3
        
        self.convblock2 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels = 3, out_channels = 32, kernel_size = (3,3), stride=2, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, groups = 16, bias=False),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 32 output_size = 32 receptive_field = 5
        

        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 32 output_size = 16 receptive_field = 10
        
        
        ## CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            # Dilated Convolution of 3
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1,dilation = 1, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        ) # input_size = 16 output_size = 16 receptive_field = 14
        
        self.convblock4 = nn.Sequential(
            # Dilated Convolution of 3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, groups = 16, bias=False),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        ) # input_size = 16 output_size = 16 receptive_field = 16
        
        self.pool2 = nn.MaxPool2d(2, 2) # input_size = 16 output_size = 8   receptive_field = 32
        
        
        ## CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, dilation = 2, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 6   output_size = 6 receptive_field = 35       
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, dilation = 1, groups = 16, bias=False),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 6   output_size = 6 receptive_field = 39 
        
        ## CONVOLUTION BLOCK 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 8   output_size = 8 receptive_field = 43
        
#         self.convblock8 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0, dilation = 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.Dropout(dropout_value),
#             nn.ReLU()
#         ) # input_size = 8   output_size = 6  receptive_field = 45
        
        
        
        self.gap = nn.AvgPool2d(kernel_size=(6,6))        
        self.fc1 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool1(self.convblock2(self.convblock1(x)))
        x = self.pool2(self.convblock4(self.convblock3(x)))
        x = self.convblock6(self.convblock5(x))
        x = self.convblock7(x)
#         x = self.convblock8(self.convblock7(x))
        x = self.gap(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        
        return F.log_softmax(x, dim=-1)
```

## LN Network

The provided network is a modified version of the LNet architecture designed for image classification on the CIFAR-10 dataset. It consists of multiple convolutional blocks followed by pooling and fully connected layers. The network is parameterized with a multiplier value that scales the number of channels in each convolutional layer.

Here's a breakdown of the network architecture:

1. **Convolution Block 1:**
   - Input: 3-channel images
   - Convolutional layer with 16*multiplier output channels, 3x3 kernel, stride 1, and padding 1
   - Group normalization with 1 group
   - Dropout with a specified dropout value
   - ReLU activation function
2. **Convolution Block 2:**
   - Convolutional layer with 16*multiplier input channels and 32*multiplier output channels, 3x3 kernel, and padding 1
   - Convolutional layer with 32*multiplier input channels and 8*multiplier output channels, 1x1 kernel
   - Group normalization with 1 group
   - Dropout with a specified dropout value
   - ReLU activation function
3. **Pooling Layer 1:**
   - Max pooling with a 2x2 kernel and stride 2
4. **Convolution Block 3:**
   - Convolutional layer with 8*multiplier input channels and 16*multiplier output channels, 3x3 kernel, padding 1, and dilation 1
   - Group normalization with 1 group
   - Dropout with a specified dropout value
   - ReLU activation function
5. **Convolution Block 4:**
   - Convolutional layer with 16*multiplier input channels and 32*multiplier output channels, 3x3 kernel, and padding 1
   - Convolutional layer with 32*multiplier input channels and 8*multiplier output channels, 1x1 kernel
   - Group normalization with 1 group
   - Dropout with a specified dropout value
   - ReLU activation function
6. **Pooling Layer 2:**
   - Max pooling with a 2x2 kernel and stride 2
7. **Convolution Block 5:**
   - Convolutional layer with 8*multiplier input channels and 16*multiplier output channels, 3x3 kernel, padding 1, and dilation 2
   - Group normalization with 1 group
   - Dropout with a specified dropout value
   - ReLU activation function
8. **Convolution Block 6:**
   - Convolutional layer with 16*multiplier input channels and 32*multiplier output channels, 3x3 kernel, padding 1, and dilation 1
   - Convolutional layer with 32*multiplier input channels and 8*multiplier output channels, 1x1 kernel
   - Group normalization with 1 group
   - Dropout with a specified dropout value
   - ReLU activation function
9. **Convolution Block 7:**
   - Convolutional layer with 8*multiplier input channels and 16*multiplier output channels, 3x3 kernel, padding 1, and dilation 1
   - Group normalization with 1 group
   - Dropout with a specified dropout value
   - ReLU activation function
10. **Global Average Pooling Layer:**
    - Average pooling layer with a kernel size of 6x6
11. **Fully Connected Layer:**
    - Linear layer with input size 16*multiplier and output size 10

### Design

```
multiplier = 2
class LNet(nn.Module):
    def __init__(self,dropout_value = 0):
        super(LNet, self).__init__()


        ## CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16*multiplier, kernel_size = (3,3), stride=1, padding=1),
#             nn.Conv2d(in_channels=3, out_channels=33, kernel_size=(3, 3), padding=1, groups = 3, bias=False),
#             nn.Conv2d(in_channels=33, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            # nn.BatchNorm2d(16),
            nn.GroupNorm(1,16*multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 32 output_size = 32 receptive_field = 3

        self.convblock2 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels = 3, out_channels = 32, kernel_size = (3,3), stride=2, padding=1),
            nn.Conv2d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=32*multiplier, out_channels=8*multiplier, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(8),
            nn.GroupNorm(1,8*multiplier),                            
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 32 output_size = 32 receptive_field = 5


        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 32 output_size = 16 receptive_field = 10


        ## CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            # Dilated Convolution of 3
            nn.Conv2d(in_channels=8*multiplier, out_channels=16*multiplier, kernel_size=(3, 3), padding=1,dilation = 1, bias=False),
            # nn.BatchNorm2d(16),
            nn.GroupNorm(1,16*multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        ) # input_size = 16 output_size = 16 receptive_field = 14

        self.convblock4 = nn.Sequential(
            # Dilated Convolution of 3
            nn.Conv2d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=32*multiplier, out_channels=8*multiplier, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(8),
            nn.GroupNorm(1,8*multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        ) # input_size = 16 output_size = 16 receptive_field = 16

        self.pool2 = nn.MaxPool2d(2, 2) # input_size = 16 output_size = 8   receptive_field = 32


        ## CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8*multiplier, out_channels=16*multiplier, kernel_size=(3, 3), padding=1, dilation = 2, bias=False),
            # nn.BatchNorm2d(16),
            nn.GroupNorm(1,16*multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 6   output_size = 6 receptive_field = 35

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            nn.Conv2d(in_channels=32*multiplier, out_channels=8*multiplier, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(8),
            nn.GroupNorm(1,8*multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 6   output_size = 6 receptive_field = 39

        ## CONVOLUTION BLOCK 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8*multiplier, out_channels=16*multiplier, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            # nn.BatchNorm2d(16),
            nn.GroupNorm(1,16*multiplier),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 8   output_size = 8 receptive_field = 43

#         self.convblock8 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0, dilation = 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.Dropout(dropout_value),
#             nn.ReLU()
#         ) # input_size = 8   output_size = 6  receptive_field = 45



        self.gap = nn.AvgPool2d(kernel_size=(6,6))
        self.fc1 = nn.Linear(16*multiplier, 10)

    def forward(self, x):
        x = self.pool1(self.convblock2(self.convblock1(x)))
        x = self.pool2(self.convblock4(self.convblock3(x)))
        x = self.convblock6(self.convblock5(x))
        x = self.convblock7(x)
#         x = self.convblock8(self.convblock7(x))
        x = self.gap(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=-1)

def model_lcifar():
    return LNet()
```

