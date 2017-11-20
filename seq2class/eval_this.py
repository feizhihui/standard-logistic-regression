# encoding=utf-8
import data_input
import tensorflow as tf
import numpy as np
import logistic_model
from sklearn import metrics

batch_size = 9600
show_step = 200

master = data_input.DataMaster(train_mode=False)

model = logistic_model.LogisticModel(on_train=False)

saver = tf.train.Saver()

y_pred = []
y_logits = []
with tf.Session() as sess:
    saver.restore(sess, '../Data/logistic_model')
    sess.run(tf.local_variables_initializer())
    for step, index in enumerate(range(0, master.datasize, batch_size)):
        batch_xs = master.datasets[index:index + batch_size]
        batch_ys = master.datalabels[index:index + batch_size]
        predictions, logits, batch_accuracy, auc = sess.run([model.preds, model.logits, model.accuracy, model.auc_opt],
                                                            feed_dict={model.input_data: batch_xs,
                                                                       model.labels: batch_ys})
        y_pred.extend(predictions.tolist())
        y_logits.extend(logits.tolist())

        if step % show_step == 0:
            print("step %d(/%d):" % (step + 1, master.datasize // (batch_size) + 1))
            print("accuracy: %.3f, auc: %.6f" % (batch_accuracy, auc))
            predictions = sess.run(model.preds, feed_dict={model.input_data: batch_xs})

    y_pred = np.array(y_pred).reshape(-1)
    y_logits = np.array(y_logits).reshape(-1)
    labels = master.datalabels.reshape(-1)

    print("==========================================")
    print("testset samples number:", labels.shape[0])
    print("eval result:")
    print("accuracy %.6f" % metrics.accuracy_score(labels, y_pred))
    print("Precision %.6f" % metrics.precision_score(labels, y_pred))
    print("Recall %.6f" % metrics.recall_score(labels, y_pred))
    print("f1_score %.6f" % metrics.f1_score(labels, y_pred))
    fpr, tpr, threshold = metrics.roc_curve(labels, y_logits)
    print("auc_socre %.6f" % metrics.auc(fpr, tpr))
