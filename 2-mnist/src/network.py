#!/usr/bin/env python

import input_data
import tensorflow as tf
import os

# Load MNIST dataset
path = '{}/../../data/mnist'.format(os.path.dirname(os.path.realpath(__file__)))
mnist = input_data.read_data_sets(path, one_hot=True)

# Define model
