import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# neural network params
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

# model path and name
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'

# define the training
def train(mnist):
    x = tf.placeholder(tf.float32, 
        [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], 
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    # init and use regularizer function
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # init the moving average class 
    variable_avg = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_avg_op = variable_avg.apply(tf.trainable_variables())

    # calculate the cross entropy of forecast (y) and actual (y_)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # define learning rate and train step
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 
        mnist.train.num_examples, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)
    # update the params and avg value in the same time
    with tf.control_dependencies([train_step, variable_avg_op]):
        train_op = tf.no_op(name='train')

    # init tensorflow model saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, 
                mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], 
                feed_dict={x: reshaped_xs, y_: ys})

            # save model every 1000 times
            if i % 100 == 0:
                print('after %d training steps, loss value is %g ' %(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 
                    global_step=global_step)

# define the main function
def main(argv=None):
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
