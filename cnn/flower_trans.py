import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# define data and model files
INPUT_DATA = './data/flower_processed.npy'
TRAIN_FILE = './model/flower'
CKPT_FILE = './model/inception_v3.ckpt'

# define training params
LEARNING_RATE = 0.0001
STEPS = 5
BATCH = 32
N_CLASSES = 5

# exclude from model and re-train
CKPT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

# load all tuned variables
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CKPT_EXCLUDE_SCOPES.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

# load trainable variables
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]

    variables_to_obtain = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_obtain.extend(variables)
    return variables_to_obtain

def main():
    # load data from file
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    num_training = len(training_images)
    print('training: %d, validation: %d, testing: %d' 
        %(num_training, len(validation_labels), len(testing_labels)))

    # define images and labels
    images = tf.placeholder(tf.float32, [None,299,299,3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)
    trainable_variables = get_trainable_variables()
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    # calculate the correctness
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # define func to load model
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, 
        get_tuned_variables(), ignore_missing_vars=True)

    # save new model after train
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print('loading tuned variables from', CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
            sess.run(train_step, feed_dict={
                images: training_images[start:end], 
                labels: training_labels[start:end]})

            if i % 2 == 0 or i+1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=1)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    images: validation_images, labels: validation_labels})
                print('validation accuracy = %.1f%%' %(validation_accuracy * 100))

            # re-assign start and end
            start = end
            if start == num_training:
                start = 0
            end = start + BATCH
            if end > num_training:
                end = num_training
        
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print('final testing accuracy = %.1f%%' %(test_accuracy * 100))

if __name__ == '__main__':
    main()
