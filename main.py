import mnist_loader
import network
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

test_data = [(, 8)]
