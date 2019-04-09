from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('MacOSX')


HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIME_STEPS = 10
TRAINING_STEPS = 1000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


def generate_data(seq):
    X = []
    y = []

    # use num of TIME_STEPS data to predict next data
    for i in range(len(seq) - TIME_STEPS):
        X.append([seq[i:i + TIME_STEPS]])
        y.append([seq[i + TIME_STEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y, is_training):
    # use multi-layer LSTM
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)])

    # use multi-layer LSTM to compose RNN
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # outputs -> [batch_size, time, HIDDEN_SIZE]
    output = outputs[:, -1, :]

    # add full connected layer to predict
    predictions = tf.contrib.layers.fully_connected(
        output, 1, activation_fn=None)
    # return predictions if not training
    if not is_training:
        return predictions, None, None
    
    # use MSE to calculate the loss
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    # optimize the loss during training
    train_op = tf.contrib.layers.optimize_loss(loss, 
        tf.train.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
    return predictions, loss, train_op


def train(sess, train_X, train_y):
    # prepare data using dataset
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    # invoke model to get the results
    with tf.variable_scope('model'):
        _, loss, train_op = lstm_model(X, y, True)
    
    # init variables
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print('train step:', i, ', loss:', l)


def test(sess, test_X, test_y):
    # prepare data using dataset
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    # invoke model to get the results
    with tf.variable_scope('model', reuse=True):
        predication, _, _ = lstm_model(X, [0.0], False)
    
    # save prediction results
    predications = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([predication, y])
        predications.append(p)
        labels.append(l)
    
    # use RMSE to calculate the error
    predications = np.array(predications).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predications - labels) ** 2).mean(axis=0))
    print('root mean square error:', rmse)

    # display as plot
    plt.figure()
    plt.plot(predications, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


def forecast_sin():
    test_start = (TRAINING_EXAMPLES + TIME_STEPS) * SAMPLE_GAP
    test_end = test_start + (TESTING_EXAMPLES + TIME_STEPS) * SAMPLE_GAP
    # generate evenly spaced numbers (start, stop, num)
    train_X, train_y = generate_data(np.sin(np.linspace(
        0, test_start, TRAINING_EXAMPLES + TIME_STEPS, dtype=np.float32)))
    test_X, test_y = generate_data(np.sin(np.linspace(
        test_start, test_end, TESTING_EXAMPLES + TIME_STEPS, dtype=np.float32)))

    with tf.Session() as sess:
        train(sess, train_X, train_y)
        test(sess, test_X, test_y)


if __name__ == '__main__':
    forecast_sin()
