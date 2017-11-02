# Standard logistic regression
This repo is a code template for standard logistic regression.<br>
The code contains the calculation methods for computing accuracy, precision, recall, f-measure and auc score using tensorflow intrinsic functions.
***
**Accuracy, Precision, Recall, F-measure, AUC score computing snippet**
```python
    print("accuracy %.6f" % metrics.accuracy_score(labels, y_pred))
    print("Precision %.6f" % metrics.precision_score(labels, y_pred))
    print("Recall %.6f" % metrics.recall_score(labels, y_pred))
    print("f1_score %.6f" % metrics.f1_score(labels, y_pred))
    fpr, tpr, threshold = metrics.roc_curve(labels, y_logits)
    print("auc_socre %.6f" % metrics.auc(fpr, tpr))
```
**AUC score computing snippet(in TensorFlow)**
```python
self.auc, self.auc_opt = tf.contrib.metrics.streaming_auc(self.logits, self.labels)
```
**Batch normalization snippet**
```python
    def batch_norm_layer(self, output, neural_size, on_train):
        # get mean,var from this mini-batch
        fc_mean, fc_var = tf.nn.moments(
            output,
            axes=[0],  # 想要 normalize 的维度, [0] 代表 batch 维度
            # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
        )
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        # mean, var = mean_var_with_update()  # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])  # defin a op: shadow_X=decay*shadow_X+(1-decay)*X
            with tf.control_dependencies([ema_apply_op]):  # 指定计算顺序(运行运算符后才能调用下列变量)
                return tf.identity(fc_mean), tf.identity(fc_var)  # return a copy

        mean, var = tf.cond(tf.constant(on_train),  # on_train 的值是 True/False
                            mean_var_with_update,  # 如果是 True, 更新 mean/var
                            lambda: (  # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                                ema.average(fc_mean),
                                ema.average(fc_var))
                            )

        # 将修改后的 mean / var 放入下面的公式
        scale = tf.Variable(tf.ones([neural_size]))
        shift = tf.Variable(tf.zeros([neural_size]))
        epsilon = 0.001
        # Y = (Y - mean) / tf.sqrt(var + epsilon)
        # Y = Y * scale + shift
        output = tf.nn.batch_normalization(output, mean, var, shift, scale, epsilon)
        return output
```
**Learning rate decay snippet**
```python
        self.global_step = tf.Variable(0)
        self.decay_lr = tf.train.exponential_decay(learning_rate,
                                                   self.global_step,
                                                   decay_steps=decay_step,
                                                   decay_rate=0.98, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.decay_lr).minimize(self.cost,
                                                                                     global_step=self.global_step)

```

