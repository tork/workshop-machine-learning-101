#!/usr/bin/env python

import input_data
import tensorflow as tf
import os



################
##### DATA #####
# Load the MNIST dataset.
# It consists of many grayscale images of handwritten digits,
# along with the corresponding numerical values they represent.
path = '{}/../../data/mnist'.format(os.path.dirname(os.path.realpath(__file__)))
mnist = input_data.read_data_sets(path, one_hot=True)



#################
##### MODEL #####
# We need to define our model. In this example, we make a simple network with
# an input and output layer. No hidden layers, that is.
#
# First we define our input layer. A placeholder is used to ensure we can
# populate the layer with MNIST input data during training.
# An MNIST-image is 28x28 pixels. If we assign each pixel to a neuron,
# we would need 28x28 = 784 neurons.
x = tf.placeholder('float', [None, 784])
# x is now a placeholder for input data of type float, with dimensions [None, 784].
# When a dimension is None, it means it will adjust to any length required.
# In other words, we can pass one 28x28 image to the placeholder, or 100 of them.
# Passing more images at the same time is called batching, and makes learning efficient.

# Weights are needed between input layer x and the output layer.
# Each image in the MNIST dataset corresponds to a digit. In other words, we
# are dealing with a classification problem with 10 classes (one class per
# digit 0-9). To model this, we need 10 output neurons. The neuron that
# outputs the highest value, is regarded as the most likely digit in an image.
# Because there is one weight for every neural connection, we would need 784x10
# weights. We store the weights in a Tensorflow variable. They are initialized
# randomly between -0.1 and 0.1.
W = tf.Variable(tf.random_uniform([784, 10], -0.1, 0.1))

# Hidden and output layers can benefit from bias neurons. There is one bias value
# for every neuron in the layer, 10 in the case of our output layer.
# The bias may be initialized to 0, and like weights they will adapt during training.
b = tf.Variable(tf.zeros([10]))

# With input x, weights W and bias b, we have everything we need to connect to
# the next and (in this case) final layer; we just need to select an activation function.
# We will use the logistic function, a kind of sigmoid function.
y = tf.nn.sigmoid(tf.matmul(x, W) + b)
# Exactly what 'tf.matmul(x, W) + b' does mathematically should not be our focus here,
# but it really is just multiplying input x by weights W, and adding bias b to it.
# The result is one value for each output neuron. We then pass this into sigmoid,
# which ensures each output neuron activates/outputs between 0 and 1.


# With the network in place, we need to define operations for training it.
# First we need a place to store the ideal training values for our dataset.
# (Ideal values, aka. target values aka. labels)
# Like the input layer, the ideal values are passed in at execution time, and
# therefore need to be referenced here by a placeholder.
# Same number of neurons as y, but scalable in first dimension, like x.
y_ = tf.placeholder('float', [None, 10])

# The learning algorithm needs a value to minimize. In this example, we simply
# use the squared difference between ideal (y_) and actual (y) values.
loss = tf.pow(y - y_, 2)
# A simple way to monitor how well the learning process is doing, is to
# look at the average error every so often.
mean_loss = tf.reduce_mean(loss)

# A learning algorithm is needed to adjust the network according to the loss.
# Tensorflow offers a range of optimizers (learning algorithms),
# including classic gradient descent implemented through backpropagation.
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# 0.01 is the learning rate.



#################
##### LEARN #####
# Create a session for learning using the CPU
sess = tf.Session()
# Run the operation for initializing variables (assigning them default values).
sess.run(tf.initialize_all_variables())

# Learn for 1000 iterations
for epoch in range(5000):
    # Retrieve batches of training samples
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # Run training of batch
    [_, loss] = sess.run([train, mean_loss],
        feed_dict={x: batch_xs, y_: batch_ys})
    # Print the average error every so often
    if not epoch % 100:
        print 'epoch #{}: {} loss'.format(epoch, loss)



################
##### TEST #####
# 'tf.argmax' returns the index of the neuron with the highest value/activation.
# In other words, it can tell us what classes the network predicted as most
# probable. We can use 'tf.equal' to see how many of the predicted classes we
# got right, compared to the ideal values in y_.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Casting correct_predictions to float, True=1.0 and False=0.0. Calculating
# the mean of these values. If we hit 90% correctly, accuracy will be 0.9.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# Testing network and returning the accuracy of the network
a = sess.run(accuracy,
    feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print 'test set accuracy: {}'.format(a)
