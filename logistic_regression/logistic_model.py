# encoding=utf-8
import data_input
import tensorflow as tf

"""
batch_size = 16
layers = [3, 16, 1]
learning_rate = 0.01
threshold = 0.5
epoch_num = 30
show_step = 200
"""

batch_size = 32
layers = [3, 16, 1]
learning_rate = 0.1
threshold = 0.5
epoch_num = 30
show_step = 200


class LogisticModel(object):
    def __init__(self):
        self.input_data = tf.placeholder(tf.float32, [None, layers[0]])
        self.labels = tf.placeholder(tf.int32, [None, 1])
        output = self.input_data
        for i in range(1, len(layers)):
            vshape = [layers[i - 1], layers[i]]
            W = tf.Variable(tf.truncated_normal(vshape) * 0.01, name="weight_" + str(i))
            b = tf.Variable(0.0, name="bias_" + str(i))
            output = tf.matmul(output, W) + b
            if i < len(layers) - 1:
                output = tf.nn.relu(output)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.labels, tf.float32), logits=output)
        self.cost = tf.reduce_mean(loss)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.logits = tf.sigmoid(output)
        self.preds = tf.where(tf.greater(self.logits, threshold), tf.ones_like(output, dtype=tf.int32),
                              tf.zeros_like(output, dtype=tf.int32))

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, self.labels), tf.float32))
        self.auc, self.auc_opt = tf.contrib.metrics.streaming_auc(self.logits, self.labels)
