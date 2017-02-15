from model import FFNN

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util

import random

class TestFFNN(test_util.TensorFlowTestCase):
    def test_and(self):
        self.assert_model(lambda x: x[0]&x[1])

    def test_xor(self):
        self.assert_model(lambda x: x[0]^x[1])

    def assert_model(self, reducer, epochs=5000):
        raw = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

        x = np.array(raw, dtype=np.float32)
        y = np.array(map(reducer, raw), dtype=x.dtype)[:,None]

        model = FFNN(2, 1)
        model.build()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            feed_dict = { model.input: x, model.ideal: y }

            for _ in xrange(0, epochs):
                sess.run(model.train, feed_dict=feed_dict)

            actual = sess.run(model.output, feed_dict=feed_dict)
            err_msg = 'actual:\n{}\nideal:\n{}'.format(actual, y)
            assert ((actual > 0.5) == y).all(), err_msg
            assert (np.absolute(actual - 0.5) > 0.3).all(), err_msg
