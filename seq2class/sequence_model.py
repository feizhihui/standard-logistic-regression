# encoding=utf-8
import tensorflow as tf
import numpy as np

n_hidden = 20
keep_prob = 0.9
lr = 0.01
sequence_lens = 41
class_num = 2


class SeqModel(object):
    def __init__(self):
        self.x = tf.placeholder(tf.int32, [None, sequence_lens])
        self.y = tf.placeholder(tf.int32, [None])

        input_x = tf.one_hot(self.x, depth=4)
        # Current data input shape: (batch_size, n_steps, n_input)
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)
        # network = rnn_cell.MultiRNNCell([lstm_fw_cell, lstm_bw_cell] * 3)
        # x shape is [batch_size, max_time, input_size]
        outputs, output_sate = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_x,
                                                               sequence_length=tf.ones_like(self.y,
                                                                                            dtype=tf.int32) * sequence_lens,
                                                               dtype=tf.float32)
        #
        # # outputs, output_sate = tf.nn.dynamic_rnn(lstm_bw_cell, x, dtype=tf.float32)
        # # shape is n*40*(n_hidden+n_hidden) because of forward + backward
        outputs = (outputs[0][:, -1, :], outputs[1][:, 0, :])
        outputs = tf.concat(outputs, 1)

        with tf.name_scope("sigmoid_layer"):
            weights = tf.Variable(tf.truncated_normal([2 * n_hidden, class_num]) * np.sqrt(2.0 / (2 * n_hidden)),
                                  dtype=tf.float32)
            bias = tf.Variable(tf.zeros([1, class_num]), dtype=tf.float32)
            logits = tf.matmul(outputs, weights) + bias
            self.activation_logits = tf.nn.sigmoid(logits)[:, 1]

        with tf.name_scope("evaluation"):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.one_hot(self.y, depth=2), logits=logits)
            self.cost = tf.reduce_mean(loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)
            self.prediction = tf.arg_max(logits, dimension=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.prediction, tf.int32), self.y), tf.float32))
            self.auc, self.auc_opt = tf.contrib.metrics.streaming_auc(self.activation_logits, self.y)
