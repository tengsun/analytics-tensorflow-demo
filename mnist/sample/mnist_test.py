import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

TEST_INTERVAL_SECS = 10

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # calculate the result
        y = mnist_inference.inference(x, None)

        # calcuate the accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # init the moving average class 
        variable_avg = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_avg.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # test model every x seconds
        while True:
            with tf.Session() as sess:
                # find the latest model in the path
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('after %s training steps, validation accuracy is %g ' 
                        %(global_step, accuracy_score))
                else:
                    print('no checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

# define the main function
def main(argv=None):
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    test(mnist)

if __name__ == '__main__':
    tf.app.run()
