from model import FFNN

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util

import random

class TestFFNN(test_util.TensorFlowTestCase):
    def test_xor(self):
        raw = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

        x = np.array(raw, dtype=np.float32)
        y = np.array(map(lambda x: x[0]^x[1], raw), dtype=x.dtype)[:,None]

        model = FFNN(2)
        model.build()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            feed_dict = { model.input: x, model.ideal: y }

            for _ in xrange(0, 10000):
                sess.run(model.train, feed_dict=feed_dict)
            actual = sess.run(model.y, feed_dict=feed_dict)
            assert ((actual > 0.5) == y).all()
