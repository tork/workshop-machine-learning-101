#!/usr/bin/env python

import sys

import tensorflow as tf
import numpy as np

import data
from model import FFNN

def main():
    # fixed seed for reproducibility
    # np.random.seed(0)

    # load dataset
    dataset_full = data.titanic()
    # partition into train, validation and test sets
    dataset_train, dataset_test = dataset_full.split(0.8)
    dataset_train, dataset_valid = dataset_train.split(0.8)

    # create network
    model = FFNN(dataset_full.nvars, dataset_full.nout)
    model.build()

    # a saver is used for storing and restoring the model during training.
    saver = tf.train.Saver()
    # the titanic dataset is tiny, and we can run through the entire thing in
    # a single batch. in other words, the data feed is constant every epoch.
    feed_train = { model.input: dataset_train.x, model.ideal: dataset_train.y }
    feed_valid = { model.input: dataset_valid.x, model.ideal: dataset_valid.y }
    feed_test = { model.input: dataset_test.x, model.ideal: dataset_test.y }

    loss_best = float('inf') # lowest validation loss achieved
    divergence_count = 0 # number of consecutive iterations with a negative result
    divergence_max = 5 # how many iterations in the wrong direction do we explore before stopping?
    validation_frequency = 250 # how many epochs between model validation?

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in xrange(0, sys.maxint):
            # learn from the entire dataset each epoch
            sess.run(model.train, feed_dict=feed_train)

            # validate model
            if not epoch % validation_frequency:
                # compute training and validation loss
                loss_train = sess.run(model.mean_loss, feed_dict=feed_train)
                loss_valid = sess.run(model.mean_loss, feed_dict=feed_valid)
                print 'epoch #{}: train={}, valid={}'.format(epoch, loss_train, loss_valid)

                # simple early stop logic.
                # if the validation loss doesn't improve after divergence_max
                # attempts, stop learning and use the best model found.
                if loss_valid < loss_best:
                    # new best, save model
                    save_path = saver.save(sess, '/tmp/model.ckpt')
                    loss_best = loss_valid
                    divergence_count = 0
                else:
                    divergence_count += 1

                if divergence_count >= divergence_max:
                    print 'no improvement in {} epochs, stopping...'\
                        .format(divergence_max * validation_frequency)
                    # restore model with best result
                    saver.restore(sess, "/tmp/model.ckpt")
                    break
        # evaluate final model on the unseen test set.
        # this is the ultimate test on how well the model generalizes.
        loss_test, actual_test = sess.run([model.mean_loss, model.output], feed_dict=feed_test)
        accuracy_test = (actual_test.round() == dataset_test.y).sum() / float(len(actual_test))
        print 'test dataset: loss={}, accuracy={}'.format(loss_test, accuracy_test)

        # evaluate your own values in test/custom.csv
        # dataset_custom = data.titanic('test/custom.csv',
        #     norm_stats=dataset_full.norm_stats,
        #     ohot_stats=dataset_full.ohot_stats,
        #     shuffle=False,
        #     read_ideal=False)
        # feed_custom = { model.input: dataset_custom.x }
        # prediction_custom = sess.run(model.output, feed_dict=feed_custom)
        # print 'predicted survival rate: {}'.format(prediction_custom)

if __name__ == '__main__':
    main()
