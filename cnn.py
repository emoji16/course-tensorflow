# -*- coding: utf-8 -*-
'''
aims:保留特征+降维(共享参数)，

CNN components：
* 卷积层 ：提取特征 stride (sum)channel kernel(filter)--可以理解为纹理模式
    * tf.keras.layers.Conv2D # 注意data_format,默认channels_last
    * input tensor: batch height width channel 
    * output tensor： batch height_sum  width_new filters
    * kernel-w ：height_kernel width_kernel channel(input通道数) filters(output通道数)
* 池化层 ：下采样 共享参数，降维，避免过拟合
    * tf.keras.layers.MaxPool2D(pool_size,strides,padding,data_format) # 注意data_format,默认channels_last
    * input tensor: batch rows cols channels
    * output tensor: batch pooled_rows pooled_cols channels
* 全连接层

+evaluation metric:MAP@3
mean average precision != precision
针对多个query求mean

+ tf.py_function(pyfunc,[tf.data1],[tf.data2])
在tfdata上应用py函数

直接利用经典预训练模型(结构、参数-迁移学习)：
model = tf.keras.applications.xxx
使用子类模型可能会有问题(batchnorm等细节call里没说明)

'''
import tensorflow as tf

# 手写简易tf2.0版本卷积层--理解即可
def con2d(x, w, b, pad, stride):
    N, H, W ,C = tf.shape(x)
    F, HH, WW, C = tf.shape(w)

    x = tf.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant')
    Hn = 1 + int((H + 2*pad - HH) / stride[0])
    Wn = 1 + int((W + 2*pad - WW) / stride[1])
    Y = tf.Variable(tf.zeros((N,Hn,Wn,F),dtype=tf.float32))

    for m in range(F):
        for i in range(Hn):
            for j in range(Wn):
                data = x[:, i*stride[0]:i*1+HH, j*stride[1]:j*1 + WW,:]
                filt = w[m,:,:,:]
                Y[:,i,j,m].assign(tf.reduce_sum(tf.multiply(data,filt),axis=(1,2,3))+b[m])
    
    return Y

# 手写简易tf2.0版本池化层--理解即可
def max_pool_forwar_naive(x,pool_size=(2,2),strides=(1,1)):
    N,H,W,C = tf.shape(x)
    h_p, w_p = pool_size
    h_s, w_s = strides
    Y = tf.zeros((N, (H-h_p) / h_s + 1, (W-w_p) / w_s + 1,C))
    Y = tf.Variable(Y)

    for i in tf.range(tf.shape(Y)[1]):
        for j in tf.range(tf.shape(Y)[2]):
            Y[:,i,j,:].assign(tf.math.reduce_max(x[:,i*h_s:i*h_s+h_p,j*w_s:i*w_s+w_p,:],\
                axis=(1,2),keepdims=False))
    return Y