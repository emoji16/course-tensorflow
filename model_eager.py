# -*- coding: utf-8 -*-
'''
利用GradientTape自动求导机制，自定义模型训练过程：
准确讲是eager execution过程
subclass model + GradientTape自定义compile+fit(epoch+batch+optimizer+loss+metric)

tf.GradientTape():persistent是否可以多次调用，watch_accessed_variables是否自动追踪可训练变量(一般True，不用手动watch)
gradient(target被微分 sources变量:model.trainable_variables)
optimizer.apply_gradients(zip) 更新梯度

tf.data.Dataset.from_tensor_slices + for epoch,for step

metric: result,reset_state
'''
import tensorflow as tf
from tensorflow.python.util.nest import _yield_value

# x = tf.constant(3.0)
# with tf.GradientTape() as g:
#     g.watch(x)  
#     y = x*x

# dy_dx = g.gradient(y,x) # 根据上下文计算一个/多个tensor的梯度
# print (dy_dx)

class MyModel(tf.keras.Model):
    def __init__(self, num_classes = 10):
        super(MyModel, self).__init__(name='my_model')
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)


# tf.data.Dataset数据分批
import numpy as np
x_train = np.random.random((1000,32))
y_train = np.random.random((1000,10))
x_val = np.random.random((200,32))
y_val = np.random.random((200,10))
x_test = np.random.random((200,32))
y_test = np.random.random((200,10))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
batch_size = 64
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_dataset = val_dataset.batch(64)

model = MyModel(num_classes=10)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

# 自定义train过程：compile + fit(loss optimizer metrics)过程
epochs = 3
for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value,model.trainable_weights) # trainable_variables/trainable_weights
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
        train_acc_metric(y_batch_train,logits)

    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %s' % (float(train_acc)))
    train_acc_metric.reset_states()

    for x_batch_val,y_batch_val in val_dataset:
        val_logits = model(x_batch_val)
        val_acc_metric(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    print('Validation acc over epoch: %s' % (float(val_acc)))
    val_acc_metric.reset_states()

    if step % 200 == 0:
        print("Train loss(for one batch) at step %s:%s" % (step,float(loss_value)))

