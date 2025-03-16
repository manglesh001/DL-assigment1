Project: Feedforward Neural Network for MNIST and Fashion-MNIST
Overview
This project implements a feedforward neural network from scratch to classify images from the MNIST and Fashion-MNIST datasets. The network is trained using various optimization algorithms, and the performance is logged using Weights & Biases (Wandb).

Key Features
Dataset Support:

Supports both MNIST and Fashion-MNIST datasets.

Data is normalized and reshaped for training.

Labels are one-hot encoded for compatibility with the loss functions.

Activation Functions:

Implements Sigmoid, ReLU, Tanh, and Identity activation functions.

Derivatives of these functions are used for backpropagation.

Weight Initialization:

Supports Random and Xavier initialization methods.

Forward Propagation:

Computes the output of the network for a given input.

Supports multiple layers with different activation functions.

Backpropagation:

Implements backpropagation to compute gradients for weight updates.

Supports Cross-Entropy and Mean Squared Error loss functions.

Optimizers:

Implements SGD, Momentum, Nesterov Accelerated Gradient (NAG), RMSProp, and Adam optimizers.

Includes support for weight decay (L2 regularization).

Training Loop:

Trains the network for a specified number of epochs.

Shuffles and splits data into mini-batches for training.

Logs training and validation metrics (loss and accuracy) to Wandb.

Validation:

Evaluates the model on a validation set after each epoch.

Logs validation loss and accuracy to Wandb.

Command-Line Arguments:

Allows customization of hyperparameters (e.g., learning rate, batch size, optimizer) via command-line arguments.

Wandb Integration:

Logs training and validation metrics to Wandb for real-time monitoring.

Tracks experiments and visualizes results.
