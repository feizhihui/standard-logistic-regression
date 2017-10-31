# Standard logistic regression
This repo is a code template for standard logistic regression.<br>
The code contains the calculation methods for computing accuracy, precision, recall, f-measure and auc score using tensorflow intrinsic functions.
***
```python
    print("accuracy %.6f" % metrics.accuracy_score(labels, y_pred))
    print("Precision %.6f" % metrics.precision_score(labels, y_pred))
    print("Recall %.6f" % metrics.recall_score(labels, y_pred))
    print("f1_score %.6f" % metrics.f1_score(labels, y_pred))
    fpr, tpr, threshold = metrics.roc_curve(labels, y_logits)
    print("auc_socre %.6f" % metrics.auc(fpr, tpr))
```
```python
self.auc, self.auc_opt = tf.contrib.metrics.streaming_auc(self.logits, self.labels)
```