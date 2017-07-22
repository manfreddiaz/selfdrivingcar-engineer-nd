import tensorflow as tf
import lenet as lenet
import matplotlib.pyplot as plot

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

batch_size = 50
epochs = 100


def preproccess_input(input):
    reshaped = tf.reshape(input, [-1, 28, 28, 1])
    return tf.pad(reshaped, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT')


def optimize(stack, y, learning_rate=1e-2):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(stack, y)
    loss = tf.reduce_mean(cross_entropy)
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)


def evaluate(stack, y):
    y_prediction = tf.nn.softmax(stack)
    prediction_match = tf.equal(tf.argmax(y_prediction, dimension=1), tf.argmax(y, dimension=1))
    return tf.reduce_mean(tf.cast(prediction_match, tf.float32))


def pipeline():
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    x_ = preproccess_input(x)
    stack = lenet.stack(x_)

    optimizer = optimize(stack, y)
    evaluator = evaluate(stack, y)

    losses = []

    writer = tf.summary.FileWriter("tmp/mnist-demo/1")
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)
        for epoch in range(epochs):
            x_data, y_data = mnist.train.next_batch(batch_size)
            session.run(optimizer, feed_dict={x: x_data, y: y_data})
            losses.append(session.run(evaluator, feed_dict={x: x_data, y: y_data}))

        session.close()

    plot.plot(losses)
    plot.show()

pipeline()


