import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# data set params (28x28, 1-10)
INPUT_NODE = 784
OUTPUT_NODE = 10

# neural network params
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

# calculate forward propagation result with ReLU activation function
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # if avg_class is None, use the weights and biases directly
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + 
                           avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# define the training process
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    # generate params for hidden layer
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # generate params for output layer
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
    # set None to not use average value of parameters
    y = inference(x, None, weights1, biases1, weights2, biases2)
    
    # define the global training steps
    global_step = tf.Variable(0, trainable=False)
    # init the moving average class 
    variable_avg = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_avg_op = variable_avg.apply(tf.trainable_variables())
    # use average value of parameters
    avg_y = inference(x, variable_avg, weights1, biases1, weights2, biases2)
    
    # calculate the cross entropy of forecast (y) and actual (y_)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # init and use regularizer function
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    # calculate the total loss as cross entropy and reg
    loss = cross_entropy_mean + regularization
    
    # define learning rate and train step
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 
        mnist.train.num_examples, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)
    # update the params and avg value in the same time
    with tf.control_dependencies([train_step, variable_avg_op]):
        train_op = tf.no_op(name='train')
    
    # calcuate the accuracy
    correct_prediction = tf.equal(tf.argmax(avg_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # start the training process
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('after %d training steps, validation accuracy is %g ' %(i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('after %d training steps, test accuracy is %g ' %(TRAINING_STEPS, test_acc))

# define the main function
def main(argv=None):
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
