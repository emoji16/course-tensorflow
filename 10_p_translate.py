# -*- coding: utf-8 -*-
'''
transformer-self-attention:兼顾上下文+并行

encoders*n + decoders*n + softmax/...
encoders(input): self-attention(input：word-embed相互依赖，后加add&normalization) + feed forward neural network(intput间没有一类，可以并行执行，后加add&layer normalization)
decoders(input + encoder-results): self-attention + encoder-decoder-attention + feed forward neural network

encoder：以QKsoftmax的形式接受其他words的信息 -> valuei / softmax(i)
* 过程:
    input-words
    s1->embeddings 
    s2-> new_emb * Wqueries Wkeys Wvalues (shared para)->  q,k,v for each word
    --self.attention layer--
    s3,s4->self.attention(q*k / sqrt(dk)--normalization+softmax)
    s5->softmax*value-vec按位相乘，保留关注值的value，削弱非相关词的value
    s6->所有softmax*value-vec加权向量加和，产生该位置的self-attention输出结果
    --self.attention layer--
* multihead-attention:多个attention-head--多组QKV，扩大自表达空间 --多softmax*value-vec矩阵合并
* positional encoding：结合位置信息 embedding with time signal = positional encoding + embedding
补充位置向量:和embed向量同一维度 加和
* 每层后也有add + layer normalization
* 各层前后补充了resnet

decoders(input + encoder-results): self-attention(inputs-previous output+ position coding(遮挡未计算好的位，设为-inf )) + encoder-decoder-attention(input-K,V) + feed forward neural network
* 每层后也有add + layer normalization，resnet
* 区别：
    * self-attention层：masker multihead attention层
        只保留previous output结果位，遮挡未output位
    * encoder-decoder层:multihead attention层
        从前一层outputs得到Q,从encoder的self-attention层里接收K,V


(nlp相关指标）对比rnn,encoder,decoder：
语义提取能力(指标WSD，词义消歧):transform > rnn~= cnn
长距离特征捕获能力(主谓语一致性检测等任务):RNN ~= transform > cnn
综合语义抽取能力（BLEU机器翻译任务）:transform > rnn~= cnn
并行计算能力：transform > cnn~= rnn

实践：使用transformer进行机器翻译
    * 数据准备
        * src-data: https://data.statmt.org/news-commentary/v14/
        * 使用tensorflow_datasets as tfds进行切割和序列化
            tf.keras.preprocessing.text.Tokenizer
            tf.keras.preprocessing.sequence.pad_sequences
        * 每个句子(序列)前后加上开始、结束token
'''
import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
# import tensorflow_datasets as tfds
import pandas as pd
# print(tf.__version__) # 2.7.0

# s1 - csv2tfrecord
# contents = pd.read_csv('./data_translate/news-commentary-v14.en-zh.tsv',error_bad_lines=False,sep='\t',header=None)
# print(contents.head()) # 320331

# train_df = contents[:280000] 
# val_df = contents[280000:] 
# print(val_df.shape)  # (26140, 2)

# with tf.io.TFRecordWriter('./data_translate/train.tfrecord') as writer:
#     for en, zh in train_df.values:
#         try:
#             feature = {
#                 # str.encode 转换为二进制str
#                 'en' :tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(en)])),
#                 'zh' :tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(zh)]))
#             }
#             example = tf.train.Example(features=tf.train.Features(feature = feature))
#             writer.write(example.SerializeToString())
#         except:
#             pass

# with tf.io.TFRecordWriter('./data_translate/val.tfrecord') as writer:
#     for en, zh in val_df.values:
#         try:
#             feature = {
#                 # str.encode 转换为二进制str
#                 'en' :tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(en)])),
#                 'zh' :tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(zh)]))
#             }
#             example = tf.train.Example(features=tf.train.Features(feature = feature))
#             writer.write(example.SerializeToString())
#         except:
#             pass

feature_description = {
    'en': tf.io.FixedLenFeature([],tf.string),
    'zh': tf.io.FixedLenFeature([],tf.string)
}

def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    return feature_dict['en'], feature_dict['zh']

train_examples = tf.data.TFRecordDataset('./data_translate/train.tfrecord').map(_parse_example)
val_examples = tf.data.TFRecordDataset('./data_translate/val.tfrecord').map(_parse_example)

for en,zh in train_examples.take(3):
    print(en)
    print(zh.numpy().decode('utf-8'))
    print('*' * 20)

# s2 - 构建词表
