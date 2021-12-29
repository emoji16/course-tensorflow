# -*- coding: utf-8 -*-
'''
metrics
1-常用损失函数：tf.keras.metrics api
update_state, reset_states, result().numpy()
e.g. tf.keras.metrics.BinaryAccuracy 类形式/ tf.keras.metrics.binary_accuracy 函数形式
* regression
tf.keras.metrics.MeanSquaredError
tf.keras.metrics.MeanAbsoluteError
kf.keras.metrics.MeanAbsolutePercentageError
tf.keras.metrics.RootMeanSquaredError
* classification
tf.keras.metrics.Accuracy
tf.keras.metrics.BinaryAccuracy
tf.keras.metrics.CategoricalAccuracy  # 要求y_true one-hot
tf.keras.metrics.SparseCategoricalAccuracy # 要求y_true 0,1,2,3
tf.keras.metrics.AUC
tf.keras.metrics.Precision
tf.keras.metrics.Recall
tf.keras.metrics.TopKCategoricalAccuracy  # 要求y_true one-hot

2-自定义metrics
基于类(适配model.fit)-tf.keras.metrics.Metric + __init__(self.add_weight),update_state,result,reset_states
基于函数

注意确定清楚y是否one-hot编码
'''

# 基于类自定义评估函数SparseCategoricalAccuracy_
import tensorflow as tf

class SparseCategoricalAccuracy_(tf.keras.metrics.Metric):
    def __init__(self, name='SparseCategoricalAccuracy', **kwargs):
        super(SparseCategoricalAccuracy_,self).__init__(name=name)
        self.total = self.add_weight(name='total',dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count',dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight = None):
        values = tf.cast(tf.equal(y_true,tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))
    
    def result(self):
        return self.count / self.total

    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)

s = SparseCategoricalAccuracy_()
s.update_state(tf.constant([2,1]), tf.constant([[0.1,0.9,0.8],[0.05,0.95,0]]))
print('Final result: ',s.result().numpy())

m = tf.metrics.SparseCategoricalAccuracy()
m.update_state(tf.constant([2,1]), tf.constant([[0.1,0.9,0.8],[0.05,0.95,0]]))
print('Final result: ',m.result().numpy())

# p2-应用于