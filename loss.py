# -*- coding: utf-8 -*-
'''
常用损失函数：tf.keras.losses api/ tf.函数实现/ np手动实现
e.g.
MSE / MeanSquaredError
BinaryCrossentropy / binary_crossentropy
CategoricalCrossentropy:one-hot
SparseCategoricalCrossentropy:0,1,2...

自定义损失函数：2 ways
2 ways：类实现, 函数实现
类实现-继承tf.keras.losses.Loss基类 + __init__ call 
函数实现-def f1 def f2 return f2

实例：subclass model(keras model也可适配) + focal loss自定义loss 实现mnist手写数字识别
'''
import tensorflow as tf

# p1-实例-类实现focal loss:对预测偏差大的增加loss权重
# class SparseFocalLoss(tf.keras.losses.Loss):
#     def __init__(self,gamma=2.0,class_num=10):
#         self.gamma = gamma 
#         self.class_num = class_num
#         super(SparseFocalLoss, self).__init__()

#     def call(self, y_true, y_pred):
#         y_pred = tf.nn.softmax(y_pred, axis = -1)
#         epsilon = tf.keras.backend.epsilon() #返回数值表达式中使用的模糊因子的值 1e-7
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1.0) # 将y_pred限制在epsilon和1.0范围内

#         y_true = tf.one_hot(y_true, depth=self.class_num) # tf.one-hot(value,depth)
#         y_true = tf.cast(y_true, tf.float32)

#         loss = - y_true * tf.math.pow(1-y_pred, self.gamma) * tf.math.log(y_pred)
#         loss = tf.math.reduce_sum(loss, axis=1)
#         return loss

# p1-实例-函数实现focal loss:
# def focal_loss(gamma = 2.0):
#     def focal_loss_fixed(y_true, y_pred):
#         y_pred = tf.nn.softmax(y_pred, axis = -1)
#         epsilon = tf.keras.backend.epsilon() #返回数值表达式中使用的模糊因子的值 1e-7
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1.0) # 将y_pred限制在epsilon和1.0范围内
#         y_true = tf.cast(y_true, tf.float32) # 已经one-hot编码过再转回去

#         loss = - y_true * tf.math.pow(1-y_pred, gamma) * tf.math.log(y_pred)
#         loss = tf.math.reduce_sum(loss, axis=1)
#         return loss
#     return focal_loss_fixed

# p2-应用
# minst dataset 28*28*1 60000train + 10000test+ focal_loss
import numpy as np
import matplotlib.pyplot as plt

# p2-data
mnist = np.load('./mnist.npz')
x_train, y_train, x_test, y_test = mnist['x_train'],mnist['y_train'],mnist['x_test'],mnist['y_test']
print(x_train.shape)

x_train, x_test = x_train/255.0, x_test/255.0 # normalization

fig, ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
fig.savefig('nums.png') 

x_train = x_train[...,tf.newaxis]  # 增加channel维(卷积) 相当于28*28 -> 28*28*1
x_test = x_test[...,tf.newaxis]
y_train = y_train[...]
y_test = y_test[...]

y_train = tf.one_hot(y_train,depth=10)
y_test=tf.one_hot(y_test,depth=10)

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

# p2-model
from keras.layers import Conv2D,Flatten,Dense
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2 = Conv2D(32, 3 ,activation='relu') # channel/filters数目,kernel,
        self.flatten = Flatten()
        self.d1 = Dense(128,activation='relu')
        self.d2 = Dense(10,activation='softmax')
    
    def call(self,x):
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

#loss
# loss_fn = tf.keras.losses.CategoricalCrossentropy()

# 函数式focal_loss
def focal_loss(gamma = 2.0):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis = -1)
        epsilon = tf.keras.backend.epsilon() #返回数值表达式中使用的模糊因子的值 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0) # 将y_pred限制在epsilon和1.0范围内
        y_true = tf.cast(y_true, tf.float32) # 已经one-hot编码过

        loss = - y_true * tf.math.pow(1-y_pred, gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss
    return focal_loss_fixed
# loss_fn = focal_loss(gamma = 2.0)

# 类式focal_loss
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self,gamma=2.0):
        self.gamma = gamma 
        super(FocalLoss, self).__init__()

    def call(self, y_true, y_pred):
        # y_pred = tf.nn.softmax(y_pred, axis = -1)
        epsilon = tf.keras.backend.epsilon() #返回数值表达式中使用的模糊因子的值 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0) # 将y_pred限制在epsilon和1.0范围内

        y_true = tf.cast(y_true, tf.float32)

        loss = - y_true * tf.math.pow(1-y_pred, self.gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss
loss_fn = FocalLoss(gamma = 2.0)

# optimizer
optimizer = tf.keras.optimizers.Adam()

# metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

# train
# 这里利用@tf.function+ train_step +test_step写法
# model.autograph里利用了在model中用@tf.function修饰call函数
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    t_predictions = model(images)
    t_loss = loss_fn(labels, t_predictions)

    test_loss(t_loss)
    test_accuracy(labels, t_predictions)

EPOCHS = 1
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images,labels in train_ds:
        train_step(images, labels)

    for test_images,test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss:{}, Accuracy: {}%, Test Loss:{}, Test Accuracy:{}%.'
    print(template.format(epoch+1, train_loss.result(),
            train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
