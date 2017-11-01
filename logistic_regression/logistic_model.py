# encoding=utf-8
import data_input
import tensorflow as tf

layers = [3, 16, 1]
learning_rate = 0.05
threshold = 0.5


class LogisticModel(object):
    def __init__(self, on_train):
        self.input_data = tf.placeholder(tf.float32, [None, layers[0]])
        self.labels = tf.placeholder(tf.int32, [None, 1])
        output = self.input_data
        for i in range(1, len(layers)):
            vshape = [layers[i - 1], layers[i]]
            W = tf.Variable(tf.truncated_normal(vshape) * 0.01, name="weight_" + str(i))
            b = tf.Variable(0.0, name="bias_" + str(i))
            output = tf.matmul(output, W)
            output = self.batch_norm_layer(output, layers[i], on_train)
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
