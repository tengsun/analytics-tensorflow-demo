# 5,5,3,16, 5x5 is filter size, convert depth 3 to 16
filter_weight = tf.get_variable('weights', [5,5,3,16], 
    initializer=tf.truncated_normal_initializer(stddev=0.1))

# totally 16 biases for depth 16
biases = tf.get_variable('biases', [16], 
    initializer=tf.constant_initializer(0.1))

# calculate the conv f-propagation result
conv = tf.nn.conv2d(input, filter_weight, strides=[1,1,1,1], padding='SAME')

# add bias at every node
biases = tf.nn.biases_add(conv, biases)
actived_conv = tf.nn.relu(biases)

# provide filter and stride params in pooling layer
pool = tf.nn.max_pool(actived_conv, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
