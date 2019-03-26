import tensorflow as tf

# convolutional neural network topology
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# layer 1, 2 and full connection nodes
CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512

# init weights for training
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, 
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

# init biases for training
def get_bias_variable(shape):
    biases = tf.get_variable('biases', shape, 
            initializer=tf.constant_initializer(0.0))
    return biases

# calculate forward propagation result
def inference(input_tensor, is_train, regularizer):
    # define the layer1 - convolution
    with tf.variable_scope('layer1-conv1'):
        con1_weights = get_weight_variable([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], None)
        con1_biases = get_bias_variable([CONV1_DEEP])
        conv1 = tf.nn.conv2d(input_tensor, con1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, con1_biases))
    
    # define the layer2 - pooling
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # define the layer3 - convolution
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = get_weight_variable([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], None)
        conv2_biases = get_bias_variable([CONV2_DEEP])
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    # define the layer4 - pooling
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # convert matrix to vector, pool_shape[0] is the batch size
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # define the layer5 - full connection
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = get_weight_variable([nodes, FC_SIZE], regularizer)
        fc1_biases = get_bias_variable([FC_SIZE])
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if is_train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    
    # define the layer6 - full connection
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = get_weight_variable([FC_SIZE, NUM_LABELS], regularizer)
        fc2_biases = get_bias_variable([NUM_LABELS])
        logit = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        
    return logit
