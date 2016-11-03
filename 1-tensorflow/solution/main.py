#!/usr/bin/env python

import tensorflow as tf
import numpy as np

def main():
    tensors()
    variables()
    placeholders()
    fill_in_the_blanks()

def tensors():
    # although the syntax can be mistaken for arrays,
    # python actually uses lists
    python_list = [1, 2, 3, 4, 5]
    # lists can be accessed by index
    assert python_list[2] == 3

    # numpy is a commonly used library for data processing.
    # pandas is another example, which also relies on numpy backend.
    numpy_array = np.array(python_list, dtype=np.int32)
    # arrays can be accessed by index
    assert numpy_array[2] == 3

    # tensors can be created from several sources, including
    # numpy arrays and python lists
    tensor0 = tf.constant(numpy_array)
    tensor1 = tf.constant(python_list)

    # we have now defined two tensor constants in tensorflow.
    # each object is a reference, similar to a file descriptor.
    # in other words, we cannot access data from the tensor directly.
    assert tensor0[2] != 3

    # in order to fetch contents of a tensor, we need tensorflow
    # to evaluate it for us in a `Session`.
    # sessions can be thought of as runtimes. by default, sessions
    # run on CPU, but also support GPU.
    sess = tf.Session()

    # the `run`-method of `Session` is used to evaluate tensors
    eval0 = sess.run(tensor0)
    # `run` returns a numpy array containing the result.
    # the following verifies that all elements of eval0
    # matches those in python_list and numpy_array respectively
    assert (eval0 == python_list).all()
    assert (eval0 == numpy_array).all()

    # sessions can infer multiple tensors in each run.
    # return values correspond to each tensor
    eval0, eval1 = sess.run([tensor0, tensor1])
    assert (eval0 == eval1).all()

    # we have seen how to create tensors from constant values,
    # but there are many other (more useful) ways to create them.
    # the following example defines a tensor that holds our two
    # tensors added together
    added = tf.add(tensor0, tensor1)
    # note that this new tensor depends on whatever values
    # lies in tensor0 and tensor1.
    # tensorflow resolves these dependencies automatically,
    # meaning we only need to pass `added` to the session
    eval_added = sess.run(added)
    assert (eval_added == [2, 4, 6, 8, 10]).all()

    # some python operators have been overloaded for tensors,
    # such as addition and multiplication
    eval_overloaded = sess.run((tensor0 + tensor1) * tensor1)
    assert (eval_overloaded == [2, 8, 18, 32, 50]).all()

def variables():
    sess = tf.Session()

    # variables are tensors that are stored within tensorflow,
    # even across different session runs. for machine learning,
    # the parameters of a model are typically defined as variables
    #
    # tensorflow needs to know what the initial values of the variable is.
    # the following initializer operation will set all elements in a variable
    # to seven
    init_op = tf.constant_initializer(value=7)
    v0 = tf.get_variable(name='v0', shape=[5], initializer=init_op)

    # before we can evaluate any variable, the session has to initialize it.
    # tf.initialize_all_variables() returns a single operation that will
    # initialize every variable that has been created.
    sess.run(tf.initialize_all_variables())
    # now we can evaluate our variable, and verify its contents
    assert (sess.run(v0) == [7] * 5).all()

def placeholders():
    sess = tf.Session()

    # by now we've seen how tensorflow keeps tabs on our tensors and variables,
    # and how a session can be used to evaluate them.
    # in order to accomplish anything useful however, we need a way to input data
    #
    # placeholders represent values that are only known during a session run.
    # consequently, their values can differ for each run.
    placeholder = tf.placeholder(tf.float32, shape=[5])
    # if we want to evaluate the placeholder, we also have to input the data it
    # should contain.
    # tensorflow uses a "feed dictionary" for this, where placeholder references
    # are used as keys, with their corresponding data as values.
    data = [1, 2, 3, 4, 5]
    feed_dict = { placeholder: data }
    eval_placeholder = sess.run(placeholder, feed_dict=feed_dict)
    assert (eval_placeholder == data).all()

    # running any operation that depends on a placeholder will require
    # input data to be fed.
    quadruple = placeholder * 4
    eval_quadruple = sess.run(quadruple, feed_dict=feed_dict)
    assert (eval_quadruple == [4, 8, 12, 16, 20]).all()
    data[0] = 0
    eval_quadruple = sess.run(quadruple, feed_dict=feed_dict)
    assert (eval_quadruple == [0, 8, 12, 16, 20]).all()

def fill_in_the_blanks():
    sess = tf.Session()

    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    op = a * b # alternatively: tf.mul(a, b)
    assert (sess.run(op) == [4, 10, 18]).all()

    c = tf.constant([2, 4, 6])
    op = tf.reduce_sum(c)
    assert sess.run(op) == 12

    a = tf.constant([
        [1],
        [2],
        [3]
    ])
    b = tf.constant([
        [1, 2, 3]
    ])
    # hint: this is matrix multiplication (aka. the "dot product")
    ideal = [
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9],
    ]
    op = tf.matmul(a, b)
    assert (sess.run(op) == ideal).all()


if __name__ == '__main__':
    main()
