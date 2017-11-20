# encoding=utf-8
import tensorflow as tf
from data_input import DataMaster
from sklearn import metrics
import numpy as np

n_hidden = 20
keep_prob = 0.9
lr = 0.005


class SeqModel(object):
    def __init__(self):
        self.x = tf.placeholder(tf.int32, [None, 41])
        self.y = tf.placeholder(tf.int32, [None])

        input_x = tf.one_hot(self.x, depth=3)
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
                                                                                            dtype=tf.int32) * 40,
                                                               dtype=tf.float32)
        # outputs, output_sate = tf.nn.dynamic_rnn(lstm_bw_cell, x, dtype=tf.float32)
        # shape is n*40*(n_hidden+n_hidden) because of forward + backward
        outputs = (outputs[0][:, -1, :], outputs[1][:, 0, :])
        outputs = tf.concat(outputs, 1)
        with tf.name_scope("softmax_layer"):
            weights = tf.Variable(tf.truncated_normal([2 * n_hidden, 3]) * np.sqrt(2.0 / (2 * n_hidden)),
                                  dtype=tf.float32)
            bias = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32)
            logits = tf.matmul(outputs, weights) + bias

        with tf.name_scope("evaluation"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
            self.cost = tf.reduce_mean(loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)
            self.prediction = tf.arg_max(tf.nn.softmax(logits), dimension=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.prediction, tf.int32), self.y), tf.float32))
            self.auc, self.auc_opt = tf.contrib.metrics.streaming_auc(self.logits, self.labels)

# ===================================

batch_size = 128
epoch_num = 70
show_step = 200

master = DataMaster()

model = SeqModel()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(epoch_num):
        print('epoch - ', str(epoch + 1))
        master.shuffle()
        for step, index in enumerate(range(0, master.datasize, batch_size)):
            batch_xs = master.train_x[index:index + batch_size]
            batch_ys = master.train_y[index:index + batch_size]
            sess.run(model.train_op, feed_dict={model.x: batch_xs, model.y: batch_ys})
            if step % show_step == 0:
                y_pred, batch_cost, batch_accuracy = sess.run(
                    [model.prediction, model.cost, model.accuracy],
                    feed_dict={model.x: batch_xs,
                               model.y: batch_ys})
                print("cost function: %.3f, accuracy: %.3f" % (batch_cost, batch_accuracy))
                print("Precision %.6f" % metrics.precision_score(batch_ys, y_pred))
                print("Recall %.6f" % metrics.recall_score(batch_ys, y_pred))
                print("f1_score %.6f" % metrics.f1_score(batch_ys, y_pred))

    # store
    saver = tf.train.Saver()
    saver.save(sess, '../Data/logistic_model')
    # print(sess.run(model.global_step))
    # print(sess.run(model.decay_lr))
