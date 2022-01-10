# -*- coding: utf-8 -*-
'''
tf.keras.layers.Layer子类进行自定义层 
class MyLayer()
1.__init__里定义初始化权重：
    initializer：tf.random_normal_initializer() /  tf.zero_initializer
    self.add_weights均可
2./也可以写入build 最后super(Linear,self).build(input_shape)--其中init build call都不用手动调用，且input_shape可以自动识别
3.call前向传播过程

实例-手动实现自定义全连接层
1.将自定义层加入序列化网络需要定义：get_config方法 
调用config = super.get_config + config.update({'units':self.units}) 字典键值形式保存参数配置
# def get_config(self):
#     config = super(MyDense, self).get_config()
#     config.update({'units':self.units})
#     return config
2.model.save 需要在声明的时候指定变量名字
3.model.load_model 需要在custom_objects中说明自定义部分 name-classname
'''
import tensorflow as tf
from tensorflow.python.ops.gradients_util import _Inputs

# p1-自定义layer:三种初始化参数方法,get_config
# 注意初始化的时候应该给变量name：w，b都这model.save容易出错
# 3种定义、初始化参数方法：
class Linear(tf.keras.layers.Layer):
    # def __init__(self,units=1,input_dim=4,**kwargs):  # init里最好注明**kwargs可变参数
    #     super(Linear, self).__init__(**kwargs)
    #     # method1：tf.random_normal_initializer 
    #     w_init = tf.random_normal_initializer()
    #     self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype='float32'),trainable= True)
    #     b_init = tf.zeros_initializer()
    #     self.b = tf.Variable(initial_value =b_init(shape=(units, ),dtype='float32'),trainable=True)
    #     # method2：self.add_weight
    #     self.w = self.add_weight(shape=(input_dim,units),
    #                             initializer='random_normal',
    #                             trainable =True)
    #     self.b = self.add_weight(shape=(units,),
    #                             initializer='zeros',
    #                             trainable =True)

    # method3:build,调用积累build(input_shape)
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],self.units),
                                initializer='random_normal',trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',trainable=True)
        super(Linear,self).build(input_shape)  # note：super.build(input_shape) / super.built = True

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        pass

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
data = iris.data  # (150,4)
label = iris.target
data = np.concatenate((data, label.reshape(150,1)),axis=-1)
np.random.shuffle(data) # 注意shuffle
labels = data[:,-1]
data = data[:,:4]

# x = tf.constant(data)
# linear_layer = Linear(units = 3)
# y = linear_layer(x)
# print(y)

# print('weight' , linear_layer.weights)
# print('non-trainble weight' , linear_layer.non_trainble_weights)
# print('weight' , linear_layer.trainble_weights)

# p2-实例：自定义全连接层 + softmax激活函数：get_config函数
# 定义全连接层 
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        self.units = units
        super(MyDense,self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True,
                                name='w')
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True,
                                name='b')
        super(MyDense, self).build(input_shape)

    def call(self,inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        # 将__init__内的量传为字典键值
        config = super(MyDense, self).get_config()
        config.update({'units':self.units})
        return config

# 函数式构建网络，需要get_config否则报错
# input (150,4) output 3分类 (150,1)
inputs = tf.keras.Input(shape=(4,))
x = MyDense(units=16)(inputs)
x = tf.nn.tanh(x)
x = MyDense(units=3)(x)
predictions = tf.nn.softmax(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

# SparseCategoricalCrossEntropy自动可以将label进行onehot转化，然后计算loss
model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(data, labels, batch_size=32, epochs=100,shuffle=True)
model.summary()


# 模型保存和加载
model.save('./model/model3.h5')  # weights应该有名字w b，否则易报错
_custom_objects = {'MyDense': MyDense} # 说明自定义部分
model_new = tf.keras.models.load_model('./model/model3.h5', custom_objects=_custom_objects)  # 应该说明自定义layer名-layer类名

#predict
y_pred = model_new.predict(data)
print(np.argmax(y_pred,axis=1))
