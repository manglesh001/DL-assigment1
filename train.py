import argparse
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

# Activation functions and derivatives
def identity(x):
    return x

def identity_derivative(x):
    return 1

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
    
# Weight Initialization
def initialize_weights(layers, method="xavier"):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        if method == "xavier":
            weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(1 / layers[i]))
        else:  # random
            weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
        biases.append(np.zeros((1, layers[i+1])))
    return weights, biases

## Q1 
def fashion_mnist_image():
    wandb.init(project="fashion-mnist", name="fashion-mnist-images")

    # Load the Fashion-MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Class names for Fashion-MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Create a figure to plot the images
    plt.figure(figsize=(10, 15))
    plt.suptitle("Fashion-MNIST DataSet", fontsize=16)

    # Plot one sample image for each class
    for i in range(len(class_names)):
        # Find the first occurrence of each class in the training data
        idx = np.where(y_train == i)[0][0]

        # Plot the image
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[idx], cmap='gray')
        plt.title(class_names[i])
        plt.axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()

    # Log the plot to Wandb
    wandb.log({"Fashion-MNIST Images": wandb.Image(plt)})

    # Show the plot
    plt.show()




# Forward Propagation
def forward_propagation(X, weights, biases, activation):
    activations = [X]
    zs = []

    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        zs.append(z)

        if activation[i] == "sigmoid":
            activations.append(sigmoid(z))
        elif activation[i] == "relu":
            activations.append(relu(z))
        elif activation[i] == "tanh":
            activations.append(tanh(z))
        elif activation[i] == "identity":
            activations.append(identity(z))

    return activations, zs

# Backpropagation
def backpropagation(y, activations, zs, weights, activation, loss_type="cross_entropy"):
    gradients_w = [None] * len(weights)
    gradients_b = [None] * len(weights)

    # Output layer error
    if loss_type == "cross_entropy":
        error = activations[-1] - y
    elif loss_type == "mean_squared_error":
        error = (activations[-1] - y) * activations[-1] * (1 - activations[-1])

    for i in reversed(range(len(weights))):
        if activation[i] == "sigmoid":
            delta = error * sigmoid_derivative(activations[i+1])
        elif activation[i] == "relu":
            delta = error * relu_derivative(activations[i+1])
        elif activation[i] == "tanh":
            delta = error * tanh_derivative(activations[i+1])
        elif activation[i] == "identity":
            delta = error * identity_derivative(activations[i+1])

        gradients_w[i] = np.dot(activations[i].T, delta)
        gradients_b[i] = np.sum(delta, axis=0, keepdims=True)

        error = np.dot(delta, weights[i].T)

    return gradients_w, gradients_b


# # Class FFNN
# class FFNN:
#     def __init__(self, layer_sizes, learning_rate=0.01):
#         self.layer_sizes = layer_sizes
#         self.learning_rate = learning_rate
#         self.weights = []
#         self.biases = []

#     # Initialize weights and biases
#         for i in range(len(layer_sizes) - 1):
#             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
#             self.biases.append(np.zeros((1, layer_sizes[i + 1])))

#   #softmax  activation function ouput layer
#     def softmax(self, x):
#         exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#         return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#   #Sigmoid activation function hidden layer
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def sigmoid_derivative(self, x):
#         return x * (1 - x)


#   #forward Pass FFNN
#     def forward(self, x):
#         self.activations = [x]
#         self.z_values = []
#         for i in range(len(self.weights)):
#             z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
#             self.z_values.append(z)
#             if i == len(self.weights) - 1:
#                 # Output layer uses softmax
#                 activation = self.softmax(z)
#             else:
#                 # Hidden layers use sigmoid
#                 activation = self.sigmoid(z)
#             self.activations.append(activation)
#       # Return the output probabilities
#         return self.activations[-1]

#     #Backward Pass FFNN

#     def backward(self, x, y):
#         m = x.shape[0]
#         self.deltas = [None] * len(self.weights)

#         # Output layer error
#         output_error = self.activations[-1] - y
#         self.deltas[-1] = output_error

#         # Backpropagate errors
#         for i in range(len(self.weights) - 2, -1, -1):
#             error = np.dot(self.deltas[i + 1], self.weights[i + 1].T)
#             self.deltas[i] = error * self.sigmoid_derivative(self.activations[i + 1])

#         # Update weights and biases
#         for i in range(len(self.weights)):
#             self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, self.deltas[i]) / m
#             self.biases[i] -= self.learning_rate * np.sum(self.deltas[i], axis=0, keepdims=True) / m

#     def train(self, x, y, epochs=10, batch_size=32):
#         for epoch in range(epochs):
#             for i in range(0, x.shape[0], batch_size):
#                 x_batch = x[i:i + batch_size]
#                 y_batch = y[i:i + batch_size]

#                 # Forward pass
#                 self.forward(x_batch)

#                 # Backward pass
#                 self.backward(x_batch, y_batch)

#             # Print loss at  every epoch
#             predictions = self.forward(x)
#             loss = -np.mean(y * np.log(predictions + 1e-10))
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# # Return the softmax probabilities for the input

#     def predict_probabilities(self, x):
#         return self.forward(x)


#  RMSPROP Optimizers

def rmsprop(weights, biases, gradients_w, gradients_b, lr, cache_w, cache_b, beta=0.99, epsilon=1e-8, weight_decay=0.0):
    for i in range(len(weights)):
        cache_w[i] = beta * cache_w[i] + (1 - beta) * (gradients_w[i] ** 2)
        weights[i] -= lr * (gradients_w[i] / (np.sqrt(cache_w[i]) + epsilon) + weight_decay * weights[i])
        cache_b[i] = beta * cache_b[i] + (1 - beta) * (gradients_b[i] ** 2)
        biases[i] -= lr * gradients_b[i] / (np.sqrt(cache_b[i]) + epsilon)
    return weights, biases, cache_w, cache_b

## MOMENTUM OPTIMIZER
def momentum(weights, biases, gradients_w, gradients_b, lr, velocity, beta=0.9, weight_decay=0.0):
    for i in range(len(weights)):
        velocity[i] = beta * velocity[i] + (1 - beta) * gradients_w[i]
        weights[i] -= lr * (velocity[i] + weight_decay * weights[i])
        biases[i] -= lr * gradients_b[i]
    return weights, biases, velocity

## Adam optimizer 

def adam(weights, biases, gradients_w, gradients_b, lr, m_w, v_w, m_b, v_b, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1, weight_decay=0.0):
    for i in range(len(weights)):
        m_w[i] = beta1 * m_w[i] + (1 - beta1) * gradients_w[i]
        v_w[i] = beta2 * v_w[i] + (1 - beta2) * (gradients_w[i] ** 2)
        m_hat_w = m_w[i] / (1 - beta1 ** t)
        v_hat_w = v_w[i] / (1 - beta2 ** t)
        weights[i] -= lr * (m_hat_w / (np.sqrt(v_hat_w) + epsilon) + weight_decay * weights[i])

        m_b[i] = beta1 * m_b[i] + (1 - beta1) * gradients_b[i]
        v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gradients_b[i] ** 2)
        m_hat_b = m_b[i] / (1 - beta1 ** t)
        v_hat_b = v_b[i] / (1 - beta2 ** t)
        biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
    return weights, biases, m_w, v_w, m_b, v_b



def log_conf_mat(y_true, y_pred, class_names, wandb_project="fashion-mnist", wandb_run_name="confusion-matrix"):

    wandb.init(project=wandb_project, name=wandb_run_name)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Fashion-MNIST Test Set', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Save the confusion matrix plot to a file
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Log the confusion matrix as an image to Wandb
    wandb.log({"confusion_matrix": wandb.Image(confusion_matrix_path)})


# Example usage
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat, 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## Nestrov Optimizer
def nesterov(weights, biases, gradients_w, gradients_b, lr, velocity, beta=0.9, weight_decay=0.0):
    for i in range(len(weights)):
        temp_velocity = beta * velocity[i]
        weights[i] -= lr * (beta * temp_velocity + (1 - beta) * gradients_w[i] + weight_decay * weights[i])
        biases[i] -= lr * gradients_b[i]
        velocity[i] = temp_velocity + (1 - beta) * gradients_w[i]
    return weights, biases, velocity



 ## SGD optimizer


def sgd(weights, biases, gradients_w, gradients_b, lr, weight_decay=0.0):
    for i in range(len(weights)):
        weights[i] -= lr * (gradients_w[i] + weight_decay * weights[i])
        biases[i] -= lr * gradients_b[i]
    return weights, biases


## NAdam optimizer

# def nadam(weights, biases, gradients_w, gradients_b, lr, m, v, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1, weight_decay=0.0):
#     for i in range(len(weights)):
#         m[i] = beta1 * m[i] + (1 - beta1) * gradients_w[i]
#         v[i] = beta2 * v[i] + (1 - beta2) * (gradients_w[i] ** 2)
#         m_hat = (beta1 * m[i] + (1 - beta1) * gradients_w[i]) / (1 - beta1 ** t)
#         v_hat = v[i] / (1 - beta2 ** t)
#         weights[i] -= lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * weights[i])
#         biases[i] -= lr * gradients_b[i] / (np.sqrt(v_hat) + epsilon)
#     return weights, biases, m, v


# Load dataset MNIST or Fashion_MNIST
def load_dataset(dataset_name):
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Invalid dataset name. Choose 'mnist' or 'fashion_mnist'.")

    # Normalize and reshape data
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # One-hot encode labels for train and test data
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return (X_train, y_train), (X_test, y_test)

# Train the network 
def train_network(X_train, y_train, X_val, y_val, config):
    np.random.seed(42)
    layers = [X_train.shape[1]] + [config.hidden_size] * config.num_layers + [10]
    activation = [config.activation] * config.num_layers + ["sigmoid"]

    weights, biases = initialize_weights(layers, config.weight_init)
    optimizer = config.optimizer

    velocity = [np.zeros_like(w) for w in weights]
    cache_w = [np.zeros_like(w) for w in weights]
    cache_b = [np.zeros_like(b) for b in biases]
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    batch_size = config.batch_size
    epochs = config.epochs
    lr = config.learning_rate

    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

        train_loss = 0
        train_correct = 0
        train_total = 0

        for i in range(0, X_train_shuffled.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            # Forward propagation
            activations, zs = forward_propagation(X_batch, weights, biases, activation)

            # Calculate training loss
            if config.loss == "cross_entropy":
                train_loss += -np.sum(y_batch * np.log(activations[-1] + 1e-8)) / len(y_batch)
            elif config.loss == "mean_squared_error":
                train_loss += np.mean(np.square(activations[-1] - y_batch))

            # Calculate training accuracy
            train_preds = np.argmax(activations[-1], axis=1)
            train_true = np.argmax(y_batch, axis=1)
            train_correct += np.sum(train_preds == train_true)
            train_total += len(y_batch)

            # Backpropagation
            gradients_w, gradients_b = backpropagation(y_batch, activations, zs, weights, activation, config.loss)

            # Update weights and biases based on optimizer
            if optimizer == "sgd":
                weights, biases = sgd(weights, biases, gradients_w, gradients_b, lr, config.weight_decay)
            elif optimizer == "adam":
                weights, biases, m_w, v_w, m_b, v_b = adam(weights, biases, gradients_w, gradients_b, lr, m_w, v_w, m_b, v_b, config.beta1, config.beta2, config.epsilon, epoch+1, config.weight_decay)
            elif optimizer == "nag":
                weights, biases, velocity = nesterov(weights, biases, gradients_w, gradients_b, lr, velocity, config.momentum, config.weight_decay)
            elif optimizer == "rmsprop":
                weights, biases, cache_w, cache_b = rmsprop(weights, biases, gradients_w, gradients_b, lr, cache_w, cache_b, config.beta, config.epsilon, config.weight_decay)
            elif optimizer == "momentum":
                weights, biases, velocity = momentum(weights, biases, gradients_w, gradients_b, lr, velocity, config.momentum, config.weight_decay)
            
           #elif optimizer == "nadam":
            #    weights, biases, m, v = nadam(weights, biases, gradients_w, gradients_b, lr, m_w, v_w, config.beta1, config.beta2, config.epsilon, epoch+1, config.weight_decay)

        # Calculate average training loss and accuracy
        train_loss /= (X_train_shuffled.shape[0] // batch_size)
        train_accuracy = train_correct / train_total

        # Validate model 
        val_activations, _ = forward_propagation(X_val, weights, biases, activation)
        if config.loss == "cross_entropy":
            val_loss = -np.sum(y_val * np.log(val_activations[-1] + 1e-8)) / len(y_val)
        elif config.loss == "mean_squared_error":
            val_loss = np.mean(np.square(val_activations[-1] - y_val))
        val_accuracy = np.mean(np.argmax(val_activations[-1], axis=1) == np.argmax(y_val, axis=1))

        # Wandb log 
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Print  train loss , accuracy and Val loss , accuracy
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return

# Main function
def main():

    parser = argparse.ArgumentParser(description="Train a feedforward neural network on MNIST or Fashion-MNIST.")
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005)
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier", choices=["random", "xavier"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, default=128)
    parser.add_argument("-a", "--activation", type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "relu"])
    args = parser.parse_args()

    # Initialize wandb 
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    # Load dataset
    (X_train, y_train), (X_test, y_test) = load_dataset(args.dataset)

    # Split training data into train and validation
    X_train, X_val = X_train[:54000], X_train[54000:]
    y_train, y_val = y_train[:54000], y_train[54000:]

    # Train the network
    train_network(X_train, y_train, X_val, y_val, args)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
