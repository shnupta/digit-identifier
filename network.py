# Handwritten digits identifying neural network.
# Code from: https://neuralnetworksanddeeplearning.com/chap1

### Libraries
#Standard
import random

# Third-party
import numpy as np


class Network(object):
    def __init__(self, sizes):
        # The list sizes contains the number of neurons in the respective layer of the network.
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Weights and biases are initialised randomly. N.B. the first layer is assumed to be an input layer and so these neurons won't have any biases.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.randon.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # Returns the output of the network if 'a' is the input
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # This trains the neural network using mini-batch stochastic gradient descent.
        # The training data is a list of tuples '(x, y)' representing the training inputs and desired outputs.

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            # A quick way of picking a random subset of data to use as our average for the gradient of the cost function
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)] 
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        # This updates the network's weights and biases by applying gradient descent using backpropagation to a single mini_batches
        # mini_batch is a list of tuples, eta is the learning rate
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # Returns a tuple representing the gradient for the cost function.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations layer by layer
        zs = [] # list to store all the z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-1+l].transpose())
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        # Return the number of test inputs for which the net outputs the correct result.
        # The output is assumed to be the index of whichever neuron in the final layer has the highest activation
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        # Return the vector of partial derivatives for the output activations
        return (output_activations - y)

### Misc funcitons
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    # derivative of sigmoid
    return sigmoid(z)*(1 - sigmoid(z))
