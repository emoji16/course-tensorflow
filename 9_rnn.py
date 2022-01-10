# -*- coding: utf-8 -*-
'''
p1-rnn models：RNN,SimpleRNN,LSTM,GRU,bidirectional rnns
p2-文本表示方法：one-hot/词袋:tf-idf textrank,基于主题：

p1-rnn models:
tf.keras.layers.RNN

tf.keras.layers.SimpleRNN
tf.keras.layers.SimpleRNNCell
usage: rnn = tf.keras.layers.RNN(
    tf.keras.layers.SimpleRNNCell(4)  # units output-dimenension
) == tf.keras.layers.SimpleRNN
# 这种写法可换成LSTMCell等，也可以写多个cell进行model堆叠

tf.keras.layers.LSTM
tf.keras.layers.LSTMCell
# 除了传递前一个状态(也是output)，增加：记忆细胞(c),遗忘门 + 输入门+ 输出门
# 有利于保存长期信息，避免(RNN上)长期信息上的梯度消失

tf.keras.layers.GRU
tf.keras.layers.GRUCell
# LSTM的简化版本(LSTM参数较多)：
* 更新门z--用h(t-1),x(t)决定h(t)中h(t-1)，h'(t)占比
* 重置门r--用h(t-1),x(t)决定从h(t-1)中保留哪部分信息m,h'(t)由[r,h(t-1)],x决定

双向bidirectional rnns:可concatenate 双向hidden states/avg/sum等操作

'''