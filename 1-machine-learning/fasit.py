import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../data/mnist", one_hot=True)

print("Finished extracting data")

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
y = tf.nn.softmax(tf.matmul(x,W))

y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print("Training")
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print("Testing")
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
