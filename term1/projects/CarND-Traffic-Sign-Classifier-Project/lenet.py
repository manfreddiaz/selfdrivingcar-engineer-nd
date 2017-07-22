import tensorflow as tf
import primitives as p


def stack(x):

    first_conv_stack = p.max_pool(p.relu(p.conv2d(x, filters_size=5, input_depth=1, num_filters=6)))

    second_conv_stack = p.max_pool(p.relu(p.conv2d(first_conv_stack, filters_size=5, input_depth=6, num_filters=16)))

    flatten_layer = tf.reshape(second_conv_stack, [-1, 400])

    first_fully_connected = p.fully_connected(flatten_layer, 400, 120)

    second_fully_connected = p.fully_connected(first_fully_connected, 120, 10)

    return second_fully_connected
