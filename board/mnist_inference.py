import tensorflow as tf

# neural network topology
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

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
def inference(input_tensor, regularizer):
    # define the layer1 variables
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = get_bias_variable([LAYER1_NODE])
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    
    # define the layer2 variables
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = get_bias_variable([OUTPUT_NODE])
        layer2 = tf.matmul(layer1, weights) + biases
    
    return layer2
