## Project: Feedforward Neural Network for MNIST and Fashion-MNIST
This project implements a feedforward neural network from scratch to classify images from the MNIST and Fashion-MNIST datasets. The network is trained using various optimization algorithms, and the performance is logged using Weights & Biases (Wandb).

Key Features
## Dataset: MNIST and Fashion-MNIST datasets.
Normalized and reshaped Dataset.
Labels one-hot encoded for compatable

## Activation Functions & its Derivaites:
Implements Sigmoid, ReLU, Tanh, and Identity activation functions.

## Weight Initialization: Random and Xavier initialization methods.

## Forward Propagation:
 Computes the output of the network for a given input.
 also supports multiple layers with different activation functions.

## Backpropagation:
Implements backpropagation to compute gradients for weight updates,Supports Cross-Entropy and Mean Squared Error loss functions.

## Optimizers:
Implements SGD, Momentum, Nesterov Accelerated Gradient (NAG), RMSProp, and Adam optimizers.
## Weight Decay-  L2 regularization

## Training Loop:
Trains the network for a specified number of epochs.
Shuffles and splits data into mini-batches for training.
Logs training and validation metrics (loss and accuracy) to Wandb.

## Validation:
Evaluates the model on a validation set after each epoch.
Logs validation loss and accuracy to Wandb.

## Hyperparameter
- number of epochs: 5, 10
- number of hidden layers:  3, 4, 5
- size of every hidden layer:  32, 64, 128
- weight decay (L2 regularisation): 0, 0.0005,  0.5
- learning rate: 1e-3, 1 e-4 
- optimizer:  sgd, momentum, nesterov, rmsprop, adam, nadam
- batch size: 16, 32, 64
- weight initialisation: random, Xavier
- activation functions: sigmoid, tanh, ReLU

## Command-Line Arguments:  Custom 
!python train.py -wp fashion_mnist -we mangleshpatidar2233-iit-madras-alumni-association

### Arguments to be supported

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

<br>

## Wandb Integration:
Logs training and validation metrics to Wandb for real-time monitoring.
Tracks experiments and visualizes results.

