import numpy as np
import tensorflow as tf

class FFNN(object):
    def __init__(self, is_train, nvars, nclasses, dtype=np.float32, dropout=None):
        super(FFNN, self).__init__()
        self.is_train = is_train
        self.nvars = nvars
        self.nclasses = nclasses;
        self.dtype = tf.as_dtype(dtype)
        self.dropout = dropout

    # defines the network in the default graph
    def build(self):
        self.build_inference()
        self.build_loss()
        if self.is_train:
            # print 'choo choo!'
            self.build_train()

    def build_inference(self):
        # create a variable for controlling dropout
        if self.dropout:
            self.keep_prob = tf.get_variable(
                'keep_prob',
                shape=[],
                initializer=tf.constant_initializer(1.0 - self.dropout),
                dtype=self.dtype)

        def build_layer(name, size, prev_layer, a=tf.nn.relu):
            with tf.variable_scope(name) as scope:
                # in the literature, certain naming conventions exist:
                # w: weights between layers
                # b: bias values for each neuron in this layer
                # a: activation function used for each neuron in this layer
                w = tf.get_variable(
                    'w',
                    shape=[prev_layer.get_shape().as_list()[1], size],
                    initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=self.dtype),
                    dtype=self.dtype)
                b = tf.get_variable(
                    'b',
                    shape=[size],
                    initializer=tf.constant_initializer(0.1, dtype=self.dtype),
                    dtype=self.dtype)

                # activation function a takes bias plus previous layer values
                # multiplied by their weights.
                layer = a(tf.matmul(prev_layer, w) + b, name=name)

                # if applicable, enable dropout
                if self.dropout != None:
                    layer = tf.nn.dropout(layer, self.keep_prob)
                return layer

        self.input = tf.placeholder(dtype=self.dtype, shape=[None, self.nvars], name='input')
        layer = build_layer('h0', 100, self.input)
        layer = build_layer('h1', 50, layer)
        self.logits = build_layer('logits', self.nclasses, layer, tf.identity)
        self.y = tf.sigmoid(self.logits, name='y')

    def build_loss(self):
        # placeholder for ideal values
        self.ideal = tf.placeholder(dtype=self.dtype, shape=self.y.get_shape(), name='ideal')

        # the learning algorithm needs a value to minimize. in this example, we simply
        # use the squared difference between ideal and actual (self.y) values
        self.squared_error = tf.pow(self.y - self.ideal, 2)
        self.loss = tf.reduce_sum(self.squared_error)

        # sparse
        # self.ideal = tf.placeholder(dtype=tf.int32, shape=[None], name='ideal')
        # self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.ideal, name='xent')
        # self.loss = tf.reduce_mean(self.xent, name='loss')

    def build_train(self):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.0005, global_step, 500, 0.96, staircase=False)
        self.train = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
