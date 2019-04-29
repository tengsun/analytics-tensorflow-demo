import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


# use slim to define LeNet-5 network
def lenet5(inputs):
    # convert inputs to 4 dim, -1 is batch, rest is image
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])

    # layer1 conv, depth=32, filter=5x5
    net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='layer1-conv')
    # layer2 max-pool, filter=2x2, stride=2
    net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')
    # layer3 conv, layer4 max-pool
    net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='layer3-conv')
    net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

    # flatten 4d to 2d for later calculation
    net = slim.flatten(net, scope='flatten')
    # layer5 and output
    net = slim.fully_connected(net, 500, scope='layer5')
    net = slim.fully_connected(net, 10, scope='output')
    return net


def train(mnist):
    # define input and network
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    y = lenet5(x)

    # define loss and train method
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy)
    learning_rate = 0.01
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # define training process
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(10000):
            xs, ys = mnist.train.next_batch(100)
            _, loss_value = sess.run([train_op, loss], 
                feed_dict={x: xs, y_: ys})
            
            if i % 100 == 0:
                print('after {0} training steps, loss is {1:.3f}'
                    .format(i, loss_value))


if __name__ == '__main__':
    mnist = input_data.read_data_sets('../../mnist/data', one_hot=True)
    train(mnist)
