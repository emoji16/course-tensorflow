# -*- coding: utf-8 -*-
'''
tf中模型构建模式：session静态图，eager execution函数式
eager execution为tf2.0特有

自动求导机制
tf.GradientTape():persistent是否可以多次调用，watch_accessed_variables是否自动追踪可训练变量
gradient(target被微分 sources变量)

'''
import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)  
    y = x*x

dy_dx = g.gradient(y,x) # 根据上下文计算一个/多个tensor的梯度
print (dy_dx)
