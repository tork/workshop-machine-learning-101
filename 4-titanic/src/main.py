#!/usr/bin/env python

import sys

import tensorflow as tf
import numpy as np

import data
from model import FFNN

def main():
    np.random.seed(0)
    dataset_full = data.titanic()
    dataset_train, dataset_test = dataset_full.split(0.8)
    dataset_train, dataset_valid = dataset_train.split(0.8)

    model = FFNN(True, dataset_full.nvars, dataset_full.nclasses)
    model.build()

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    feed_train = { model.input: dataset_train.x, model.ideal: dataset_train.y }
    feed_valid = { model.input: dataset_test.x, model.ideal: dataset_test.y }
    feed_test = { model.input: dataset_test.x, model.ideal: dataset_test.y }

    loss_best = float('inf')
    divergence_count = 0
    divergence_max = 5
    validation_frequency = 250

    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in xrange(0, sys.maxint):
            sess.run(model.train, feed_dict=feed_train)

            # validate model
            if not epoch % validation_frequency:
                loss_train = sess.run(model.loss, feed_dict=feed_train)
                loss_valid = sess.run(model.loss, feed_dict=feed_valid)
                print 'epoch #{}: train={}, valid={}'.format(epoch, loss_train, loss_valid)

                # simple early stop logic
                if loss_valid < loss_best:
                    save_path = saver.save(sess, '/tmp/model.ckpt')
                    loss_best = loss_valid
                    divergence_count = 0
                else:
                    divergence_count += 1

                if divergence_count >= divergence_max:
                    print 'no improvement in {} epochs, stopping...'\
                        .format(divergence_max * validation_frequency)
                    saver.restore(sess, "/tmp/model.ckpt")
                    break
        loss_test = sess.run(model.loss, feed_dict=feed_test)
        print 'test loss: {}'.format(loss_test)

        # evaluate your own values in data/titanic/custom.csv
        # dataset_custom = data.titanic('data/titanic/custom.csv', norm_stats=dataset_full.norm_stats, ohot_stats=dataset_full.ohot_stats, shuffle=False)
        # feed_custom = { model.input: dataset_custom.x, model.ideal: dataset_custom.y }
        # prediction_custom = sess.run(model.y, feed_dict=feed_custom)
        # print 'predicted survival rate: {}'.format(prediction_custom)

if __name__ == '__main__':
    main()
