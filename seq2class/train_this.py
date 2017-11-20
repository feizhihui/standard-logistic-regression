# encoding=utf-8
import tensorflow as tf
from data_input import DataMaster
from sklearn import metrics
from sequence_model import SeqModel

batch_size = 128
epoch_num = 7
show_step = 200

# ===================================

master = DataMaster()
model = SeqModel()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(epoch_num):
        print('========== epoch - ', str(epoch + 1), "===================")
        master.shuffle()
        for step, index in enumerate(range(0, master.datasize, batch_size)):
            batch_xs = master.train_x[index:index + batch_size]
            batch_ys = master.train_y[index:index + batch_size]
            sess.run(model.train_op, feed_dict={model.x: batch_xs, model.y: batch_ys})
            if step % show_step == 0:
                y_pred, batch_cost, batch_accuracy, auc = sess.run(
                    [model.prediction, model.cost, model.accuracy, model.auc_opt],
                    feed_dict={model.x: batch_xs,
                               model.y: batch_ys})
                print("cost function: %.3f, accuracy: %.3f, auc: %.3f" % (batch_cost, batch_accuracy, auc))
                print("Precision %.6f" % metrics.precision_score(batch_ys, y_pred))
                print("Recall %.6f" % metrics.recall_score(batch_ys, y_pred))
                print("F1-score %.6f" % metrics.f1_score(batch_ys, y_pred))

    # store
    saver = tf.train.Saver()
    saver.save(sess, '../Data/sequence_model')
