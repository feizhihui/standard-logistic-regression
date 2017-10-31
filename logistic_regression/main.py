# encoding=utf-8
import data_input
import tensorflow as tf
import logistic_model

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

master = data_input.DataMaster()

model = logistic_model.LogisticModel()
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
                batch_cost, batch_accuracy, auc = sess.run([model.cost, model.accuracy, model.auc_opt],
                                                           feed_dict={model.input_data: batch_xs,
                                                                      model.labels: batch_ys})
                print("cost function: %.3f, accuracy: %.3f, auc: %.3f" % (batch_cost, batch_accuracy, auc))

    # store
    saver = tf.train.Saver()
    saver.save(sess, '../Data/logistic_model')
