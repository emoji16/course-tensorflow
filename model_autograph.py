# -*- coding: utf-8 -*-
'''
计算图三种构建方式：静态图-session，动态图-eager execution，AutoGraph-@tf.function装饰器
静态计算图：程序编译执行的时候生成神经网络中的结构(C++)，再执行相关操作。允许编译器进行更多优化，难debug
动态计算图：可以按照编写命令执行
AutoGraph：将动态图eager execution方式转换为静态图执行，coding + perform efficiency

AutoGraph使用三项注意:
* @tf.function修饰函数中尽量使用tf函数而不是python函数 tf.random.normal  np.random.randn 
  -- 未修饰时eager执行，每次都执行;修饰后静态图执行，只在创建时执行(python函数不能嵌入到静态计算图)
* @tf.function修饰函中不能定义tf.Variable
  -- 未修饰时eager执行，每次重新定义;修饰后静态图执行，只在创建时进行定义。tf报错
* 避免在其中修改外部的python列表或字典等结构型变量
  -- 未修饰时eager执行，每次都修改;修饰后静态图执行，只在创建时进行修改(python函数不能嵌入到静态计算图，执行无法修改)

AutoGraph详解：
@tf.function(autograph=True) 将函数功能分为两步：
* 创建计算图(python-print等,最先执行且只执行一次)
* 执行计算图(tf.print等，每次函数调用都执行)

AutoGraph封装(其中不能定义tf.Variable-->需要接口)：
* 定义类 ：tf.Variable创建放在类的初始化方法中；类的逻辑放在其他方法中
* tf.Module是tf中的基类 (tf.keras.Model也是其子类)
  -- @tf.function(input_signature=[tf.TensorSpec(shape=[],dtype=tf.float32)])
  -- 利用tf.module进行model(module)读取和加载:tf.saved_model.save/tf.saved_model.load
'''
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph

# part1-autograph tf.function执行机制
@tf.function(autograph=True) 
def myadd(a,b):
    for i in tf.range(3):
        tf.print(i)
    c = a+b
    print("tracing")
    return c

print(myadd(tf.constant("hello "), tf.constant("world"))) # tf类型变量做参数，函数第一次调用：创建+执行
print(myadd(tf.constant("hello "), tf.constant("Erin")))  # 同类型tf类型变量做参数，之后调用：仅执行
print(myadd(tf.constant(3), tf.constant(3)))  #不同类型变量做参数：创建+执行

print(myadd("hello ","world"))  #传入非tf数据类型则每次创建+执行
print(myadd("hello ","Erin"))  

# part2-tf.Module 作基类封装
class DemoModule(tf.Module):
    def __init__(self, init_value= tf.constant(0.0), name=None):
        super(DemoModule, self).__init__(name=name)
        with self.name_scope:  # == with tf.name_scope("demo_model")
            self.x = tf.Variable(init_value, dtype=tf.float32,trainable=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[],dtype=tf.float32, name='a')])
    def addprint(self, a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return (self.x)

demo = DemoModule(init_value = tf.constant(1.0))
result = demo.addprint(tf.constant(5.0))

print(demo.variables)
print(demo.trainable_variables)

print(demo.submodules)  # tf.module专为tf2.0提出

# 利用tf.module进行model(module)读取和加载:saved_model.save/saved_model.load
tf.saved_model.save(demo,'./model/demo_model/',signatures={'serving_default':demo.addprint}) # 指定跨平台部署和调用的方法
demo2 = tf.saved_model.load('./model/demo_model/')
demo2.addprint(tf.constant(5.0))

# 可以通过命令行查看,利用tf指令
# saved_model_cli show --dir ./data/ --all

# part3-autograph 实例
# class MyModel(tf.keras.Model)定义模型结构+ @tf.function装饰器实现静态图图构建
import numpy as np
class MyModel(tf.keras.Model):  # tf.keras.Model是tf.Module子类
    def __init__(self, num_classes=10):
        super(MyModel,self).__init__(name='my_model')
        self.num_classes = num_classes
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(num_classes)
    
    @tf.function(input_signature=[tf.TensorSpec([None,32], tf.float32,name='inputs')])
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

tf.saved_model.save(model,'./model/model1/',signatures={'serving_default':model.call}) 
model2 = tf.saved_model.load('./model/model1/')
f = model2.signatures["serving_default"]
# x_test = np.random.random((200,32))
a = x_test.tolist()[0] # 32
b = f(inputs = tf.constant([a]))
print(b)


