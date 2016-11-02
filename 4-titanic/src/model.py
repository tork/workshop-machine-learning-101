import numpy as np
import tensorflow as tf

class FFNN(object):
    def __init__(self, nvars):
        super(FFNN, self).__init__()
        self.nvars = nvars

    # defines the network in the default graph
    def build(self):
        pass
        ##############
        ### TASK 2 ###
        ##############
        # define your network!
        # you can try to create it by memory, but it might be appropriate
        # to copy the network you defined for mnist over.
        #
        # the number of input variables is stored in self.nvars.
        #
        ### regarding the output layer
        # this network only has one output neuron: whether the passenger
        # survives or not (the mnist net had 10 output neurons).
        #
        # since we have only one output neuron, we can't use softmax on the
        # final layer: softmax ensures the output sums to one, meaning our
        # single neuron would always output 1.0.
        #
        # hint:
        # the ideal values are 0 or 1, so perhaps the output should also lie
        # in the 0..1 range? look up the sigmoid activation function.
        #
        ### regarding exposed properties
        # the code expects three properties to be defined:
        # self.y: the output layer
        # self.loss: the loss (some sort of difference between actual and ideal output)
        # self.train: an operation for training the network (aka. optimizer)
        #
        # in the mnist example, we used cross entropy as our loss function and
        # gradient descent as training operation.
        # as mentioned, this problem is slightly different because we only have
        # one output. the loss function can simply be the difference between
        # ideal and actual output, squared and summed.
        # self.loss = (actual - ideal) ^ 2 # this is pseudocode, ^ is actually xor in python
        # self.train = gradient descent
        #
        ### does it work?
        # try running the test: python -m unittest discover <src-directory>
        # the test tries to learn from the xor operator. this is a simple
        # task, and you should use a higher learning rate for the test
        # than the titanic dataset.
        # I've been using 0.01 for xor, 0.0005 for titanic. it depends on
        # your network, variables and preprocessing.
