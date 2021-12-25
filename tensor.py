# -*- coding: utf-8 -*-

'''
part1:tensor & basic operations
tensor-tf.Variable, tf.rank, tf.shape, tf.constant, tf.zeros, tf.ones
operations-tf.strings, tf.debugging, tf.dtypes, tf.math, tf.random, tf.feature_column

part2:layers
tf.keras.layers-Dense,Conv2D,LSTM,BatchNormalization,Dropout
tf.nn-更底层

part3:modeling-build (3 methods)+ train:compile(optimizer,loss,metrics),fit(data,labels,epochs,batch_size),evaluate,predict
sequential model:层级结构-tf.keras.Sequential-model.add、tf.keras.Sequential([])
functional model:构建DAG图,可实现非常规结构(多输入,多输出),共享图层,非顺序数据流(resnet)-tf.keras.Input + tf.keras.Model(inputs=inputs, outputs=predictions)
subclassing model:子类化tf.keras.Model+自定义前向传播模型+结合eager execution-class MyModel(tf.keras.Model) + init,call
'''

import tensorflow as tf
print(tf.__version__)

print(tf.test.is_gpu_available())

# p1-tensor
mammal = tf.Variable("Elephant", tf.string)
tf.print(tf.rank(mammal))
tf.print(tf.shape(mammal))
print(tf.rank(mammal))

tf.constant([1,2,3], dtype = tf.int16)

tf.zeros((2,2), dtype = tf.int16)

tf.ones((2,2), dtype = tf.int16)

# p2-operations 
# tf.strings, tf.debugging, tf.dtypes, tf.math, tf.random, tf.feature_column

# tf.strings
tf.strings.bytes_split("hello")

# help(tf.strings.split)
tf.strings.split("hello world")

tf.strings.to_hash_bucket(["hello","world"], num_buckets=10)


# tf.random
# help(tf.random.uniform)
a=tf.random.uniform(shape=(10,5),minval=0,maxval=10)

# tf.debugging
a=tf.random.uniform((10,10))
tf.debugging.assert_equal(x=a.shape,y=(10,10))
# tf.debugging.assert_equal(x=a.shape,y=(20,10))

# tf.math
a=tf.constant([[1,2],[3,4]])
b=tf.constant([[5,6],[7,8]])
tf.print(tf.math.add(a,b))
tf.print(tf.math.subtract(a,b))
tf.print(tf.math.multiply(a,b))  # dot-product
tf.print(tf.math.divide(a,b))

# ty.dtypes
x=tf.constant([1.8,2.2],dtype=tf.float32)
x_int = tf.dtypes.cast(x,tf.int32)
print(x_int)  # tf.Tensor([1 2], shape=(2,), dtype=int32)

# p3-modeling methods:
# sequential model 1-model.add
from tensorflow.keras import layers
# model = tf.keras.Sequential()
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10))

# sequential model 2-Sequential([])
# model = tf.keras.Sequential([
#     layers.Dense(64,activation='relu',input_shape=(32,)),
#     layers.Dense(64,activation='relu'),
#     layers.Dense(10)
# ])

# functional model 
# input1=tf.keras.Input(shape=(32,))  # multi-input possible
# input2=tf.keras.Input(shape=(32,))  # multi-input possible
# x1=layers.Dense(64,activation='relu')(input1)
# x2=layers.Dense(64,activation='relu')(input2)
# x=tf.concat([x1,x2],axis=1) # [2,3]->[2,6]
# x=layers.Dense(64,activation='relu')(x)
# predictions = layers.Dense(10)(x)

# model = tf.keras.Model(inputs=[input1,input2], outputs=predictions)

# subclassing model 
class MyModel(tf.keras.Model):

    def __init__(self,num_classes=10):  # 定义model结构
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.dense_1 = layers.Dense(32,activation='relu')
        self.dense_2 = layers.Dense(num_classes)

    def call(self,inputs):  # 定义前向传播过程
        x = self.dense_1(inputs)
        return self.dense_2(x)

model = MyModel(num_classes=10)

# train
# model compile
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# model fit 
import numpy as np

data = np.random.random((1000,32))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
labels = np.random.random((1000,10))
model.fit(data, labels, epochs=10,batch_size=32) # optional para:validation_data
# model.fit((data1,data2), labels, epochs=10,batch_size=32)