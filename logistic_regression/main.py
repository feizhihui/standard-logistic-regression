# encoding=utf-8
import data_input
import tensorflow as tf
import logistic_model
from sklearn import metrics

"""
batch_size = 16
layers = [3, 16, 1]
learning_rate = 0.01
threshold = 0.5
epoch_num = 30
show_step = 200
"""

batch_size = 128
epoch_num = 60
show_step = 200

master = data_input.DataMaster()

model = logistic_model.LogisticModel(on_train=True)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    for epoch in range(epoch_num):
        print('epoch - ', str(epoch + 1))
        master.shuffle()
        for step, index in enumerate(range(0, master.datasize, batch_size)):
            batch_xs = master.train_x[index:index + batch_size]
            batch_ys = master.train_y[index:index + batch_size]
            sess.run(model.train_op, feed_dict={model.input_data: batch_xs, model.labels: batch_ys})
            if step % show_step == 0:
                y_pred, batch_cost, batch_accuracy, auc = sess.run(
                    [model.preds, model.cost, model.accuracy, model.auc_opt],
                    feed_dict={model.input_data: batch_xs,
                               model.labels: batch_ys})
                print("cost function: %.3f, accuracy: %.3f, auc: %.3f" % (batch_cost, batch_accuracy, auc))
                print("Precision %.6f" % metrics.precision_score(batch_ys, y_pred))
                print("Recall %.6f" % metrics.recall_score(batch_ys, y_pred))
                print("f1_score %.6f" % metrics.f1_score(batch_ys, y_pred))

    # store
    saver = tf.train.Saver()
    saver.save(sess, '../Data/logistic_model')
