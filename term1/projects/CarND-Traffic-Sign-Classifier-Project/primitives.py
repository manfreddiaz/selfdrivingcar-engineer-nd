import tensorflow as tf


def weights(shape, stddev=0.05):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))


def bias(shape, value=0.05):
    return tf.Variable(tf.constant(value=value, shape=shape))


def conv2d(input, filters_size, num_filters, input_depth, padding='VALID', strides=[1, 1, 1, 1]):

    W = weights([filters_size, filters_size, input_depth, num_filters])  # filters
    b = bias([num_filters])  # bias

    return tf.nn.conv2d(input, W, padding=padding, strides=strides) + b


def relu(layer):
    return tf.nn.relu(layer)


def max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
    return tf.nn.max_pool(layer, ksize=ksize, strides=strides, padding=padding)


def fully_connected(x, num_inputs, num_outputs):
    W = weights([num_inputs, num_outputs])
    b = bias([num_outputs])
    return tf.matmul(x, W) + b