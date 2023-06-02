# Introduction to PyTorch & Network Building


## Architecture Document: Neural Network
The following is the architecture document for the given neural network implemented using the PyTorch library.

### Network Architecture
The Network class represents a convolutional neural network (CNN) with multiple layers. It consists of two convolutional layers (conv1 and conv2) followed by three fully connected layers (fc1, fc2, and out). The network takes input images and produces output predictions.

### Layers and Parameters
- conv1: Convolutional layer with 1 input channel, 6 output channels, and a kernel size of 5. Receptive field: 5x5.
- conv2: Convolutional layer with 6 input channels, 12 output channels, and a kernel size of 5. Receptive field: 9x9.
- fc1: Fully connected layer with 192 input features and 120 output features.
- fc2: Fully connected layer with 120 input features and 60 output features.
- out: Fully connected layer with 60 input features and 10 output features.

### Network Diagram
Here is a diagram illustrating the architecture of the neural network:
        Input
         |
     +---v---+
     | conv1 |
     +---|---+
         v
     +---v---+
     | conv2 |
     +---|---+
         v
    +----v----+
    | Flatten |
    +----|----+
         v
    +----v----+
    |   fc1   |
    +----|----+
         v
    +----v----+
    |   fc2   |
    +----|----+
         v
    +----v----+
    |   out   |
    +---------+
         |
      Output

### Forward Propagation
The forward method defines the forward propagation logic of the neural network. It performs the following operations in sequence:

- Receives the input tensor t.
- Passes t through conv1 layer, applies ReLU activation, and performs max pooling with a kernel size of 2 and stride of 2.
- Passes the output through conv2 layer, applies ReLU activation, and performs max pooling with a kernel size of 2 and stride of 2.
- Reshapes the output tensor to a flattened shape.
- Passes the flattened tensor through fc1 layer and applies ReLU activation.
- Passes the output through fc2 layer and applies ReLU activation.
- Passes the output through the out layer.
- Returns the output tensor.
Note: The activation function used in the network is the rectified linear unit (ReLU).

## Data Loader Utils

### Data Loading function
This function does not accept any parameters and it returns a data loader object containing the training set of the FashionMNIST dataset.
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

### Image Display function
This function is used to display a batch of images along with their corresponding labels from a data loader. The input is data loader object that contains the images and labels.
    - Retrieve the next batch of images and labels from the data loader using next(iter(loader)).
    - Assign the images and labels to the variables images and labels, respectively.
    - Create a grid of images using torchvision.utils.make_grid(images, nrow=10). The nrow parameter specifies the number of images to display in each row of the grid.
    - Display the grid of images using plt.imshow(np.transpose(grid, (1, 2, 0))). The grid is transposed to match the expected image dimensions.
    - Print the corresponding labels using print('labels:', labels
