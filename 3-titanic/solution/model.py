import numpy as np
import tensorflow as tf

class FFNN(object):
    def __init__(self, nvars, nout):
        super(FFNN, self).__init__()
        self.nvars = nvars
        self.nout = nout

    def build(self):

        self.input = x = tf.placeholder('float', [None, self.nvars])
        W = tf.Variable(tf.random_uniform([self.nvars, 1]))
        b = tf.Variable(tf.zeros([1]))
        self.output = y = tf.sigmoid(tf.matmul(x, W) + b)
        self.ideal = y_ = tf.placeholder('float', [None, 1])
        loss = squared_error = tf.pow(y - y_, 2)
        self.mean_loss = mean_error = tf.reduce_mean(squared_error)
        self.train = train = tf.train.GradientDescentOptimizer(0.1).minimize(squared_error)

        ##############
        ### TASK 1 ###
        # Define your network! Hint: Adapt the model code from the MNIST problem.
        # The same network can be used here, as long as you get the layer sizes right.
        # You only need to define the network and training operations;
        # loading data and training the model is taken care of in main.py.
        #
        # The network will model whether or not a passenger survived Titanic.
        #
        # Predefined properties:
        # self.nvars: the number of input variables
        # self.nout: the number of output neurons
        #
        # Some properties need to be set, in order for the code to work:
        # self.input: placeholder for input data
        # self.output: the output layer
        # self.mean_loss: the mean loss (mean distance between actual and ideal output)
        # self.train: an operation for training the network (aka. optimizer)
        # self.ideal: placeholder for ideal data
        # A property can be set like so: self.some_number = 123
        #
        # Want to verify that your implementation works?
        # Try running the test: python -m unittest discover <src-directory>
        # The test tries to learn from the AND and XOR operators.
        # Any network should ideally be able to learn AND.
        # Only networks with one or more hidden layers are able to learn XOR.
        # We recommend using a learning rate of 0.1 for the tests.
        #
        # Once the AND-test is working, try learning from the Titanic dataset:
        # ./src/main.py

        ##############
        ### TASK 2 ###
        # With a working network in place, try getting the XOR-test to run by
        # adding a hidden layer. A hidden layer is created similarly
        # to the output layer, with it's own weights and bias. The data should
        # then flow from the input layer, through the hidden,
        # before reaching the output layer.
        # Hint: Use tf.nn.relu as the hidden layer activation function.
        # The hidden layer network should perform similarly to the regular one,
        # when applied to the Titanic dataset.

        ##############
        ### TASK 3 ###
        # Try adjusting the learning rate, and see how it impacts learning.
        # Because neural nets are initialized randomly, you should probably
        # train about 10 models and take the average accuracy for it to mean
        # something.
        # If 10 samples takes too long, feel free to use a lower number.
        # Start training by running ./src/main.py

        ##############
        ### TASK 4 ###
        # What input variables does the model pick up on? Are there some that
        # don't matter? Try to comment out one variable at a time, to see
        # how it affects the networks ability to learn. The variables can be
        # found in src/data.py, around line 19.

        ##############
        ### TASK 5 ###
        # What does your network think of your own chances of surviving Titanic?
        # Enter your own details in test/custom.csv.
        # Here is an example:
        # "pclass","sex","age","sibsp","parch"
        # 1,male,26,1,2
        # Add as many rows as you wish.
        # In src/main.py, uncomment lines 75 through 82.
        # Start training the model, and read your survival chances!
        # 1.0 = live to tell the tale, 0.0 = dead as a dodo.

        #############
        ### BONUS ###
        # See if you can improve the accuracy by making other changes to the
        # network. Things to try:
        # Adjust the number of hidden neurons
        # Use a different optimizer (other than gradient descent)
        # See how adding more layers affects learning
