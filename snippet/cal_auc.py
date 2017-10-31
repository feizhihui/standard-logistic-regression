# encoding=utf-8

import tensorflow as tf

logits = tf.Variable([0.1, 0.5, 0.0])
labels = tf.Variable([0, 1, 0], dtype=tf.int32)  # true or false

auc, opt = tf.contrib.metrics.streaming_auc(logits, labels)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())  # try commenting this line and you'll get the error
train_auc_a = sess.run([auc, opt])
train_auc_b = sess.run([auc, opt])
print(train_auc_a)
print(train_auc_b)
