# -*- coding: utf-8 -*-
'''
实践：使用transformer进行机器翻译
    * s1-数据准备
        * src-data: https://data.statmt.org/news-commentary/v14/
        * 还是用tf.keras.preprocessing进行切割和字典的构建
            tf.keras.preprocessing.text.Tokenizer
            tf.keras.preprocessing.sequence.pad_sequences
        * 每个句子序列化(jieba+nltk)+前后加上开始、结束token
        * 过滤长文本train_ds.filter.batch.shuffle.prefetch
    * s2-手写model:transformer
        * scaled dot product attention

'''
import os
import io
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.ops.gen_experimental_dataset_ops import experimental_assert_next_dataset
import tensorflow_datasets as tfds
import pandas as pd
# print(tf.__version__) # 2.7.0

# s1 - p1 - csv2tfrecord
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

# for en,zh in train_examples.take(3):
#     print(content_cut_en(en.numpy().decode('utf-8')))
#     print(content_cut_zh(zh.numpy().decode('utf-8')))
#     print('*' * 20)

# s1 - p2 - 构建中英文字典:jieba nltk分词+ tf.keras.preprocessing.text.Tokenizer序列化
output_dir = './data_translate/vocab'
en_vocab_file = os.path.join(output_dir,'en_vocab.json')
zh_vocab_file = os.path.join(output_dir,'zh_vocab.json')
checkpnt_file = os.path.join(output_dir,'checkpnt')
log_file = os.path.join(output_dir,'log')

en_vocab_size = 100000
zh_vocab_size = 100000

import jieba
def content_cut_zh(x):
    x = jieba.lcut(x)
    x = ' '.join(x)
    return x

import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
def content_cut_en(x):
    x = word_tokenize(x)
    x = ' '.join(x)
    return x

if not os.path.exists(output_dir):
    print("重新建立字典...")
    os.makedirs(output_dir)
    tokenizer_en = tf.keras.preprocessing.text.Tokenizer(num_words=en_vocab_size,lower=False,filters="")
    tokenizer_en.fit_on_texts([content_cut_en(en.numpy().decode('utf-8')) for en, _ in train_examples])
    tokenizer_en_json = tokenizer_en.to_json()
    with io.open(en_vocab_file,'w') as f:
        f.write(json.dumps(tokenizer_en_json, ensure_ascii=False))

    tokenizer_zh = tf.keras.preprocessing.text.Tokenizer(num_words=zh_vocab_size,lower=False,filters="")
    tokenizer_zh.fit_on_texts([content_cut_zh(zh.numpy().decode('utf-8')) for _, zh in train_examples])
    tokenizer_zh_json = tokenizer_zh.to_json()
    with io.open(zh_vocab_file,'w') as f:
        f.write(json.dumps(tokenizer_zh_json, ensure_ascii=False))

print("读取英文字典")
with open(en_vocab_file,'r') as f1:
    tokenizer_en = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f1))
print("读取中文字典")
with open(zh_vocab_file,'r') as f2:
    tokenizer_zh = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f2))

# s1 - p3 - 建立序列,并用tf.py_function修饰：前后加上BOS(en_vocab_size+1)+EOS(en_vocab_size+2)
# print(tokenizer_en.texts_to_sequences([content_cut_en('In any case, let’s wish him luck.')]))
# print(tokenizer_zh.texts_to_sequences([content_cut_zh('无论如何，让我们祝他好运。')]))
def encode(en_t, zh_t):
    en_indices = [en_vocab_size+1] + tokenizer_en.texts_to_sequences([content_cut_en(en_t.numpy().decode('utf-8'))])[0] + [en_vocab_size+2]
    zh_indices = [zh_vocab_size+1] + tokenizer_zh.texts_to_sequences([content_cut_zh(zh_t.numpy().decode('utf-8'))])[0] + [zh_vocab_size+2]
    #print(en_t.numpy().decode('utf-8'), en_indices)
    #print(zh_t.numpy().decode('utf-8'), zh_indices)
    return en_indices, zh_indices

# en_t ,zh_t = next(iter(train_examples))
# en_indices, zh_indices = encode(en_t, zh_t)

def tf_encode(en_t,zh_t):
    return tf.py_function(encode,[en_t,zh_t],[tf.int64, tf.int64])

# s1 - p4 - 过滤过长文本(optional：遍历cnt得到130858 examples left)
MAX_LENGTH = 40
def filter_max_length(en, zh, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(en) <= max_length, tf.size(zh) <= max_length)

# s1 - p5 - 训练集train_ds构建+分批
BATCH_SIZE = 64
BUFFER_SIZE = 15000
train_ds = train_examples.map(tf_encode)\
            .filter(filter_max_length)\
            .shuffle(BUFFER_SIZE)\
            .padded_batch(BATCH_SIZE,padded_shapes=([-1],[-1]))\
            .prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_examples.map(tf_encode)\
            .filter(filter_max_length)\
            .padded_batch(BATCH_SIZE,padded_shapes=([-1],[-1]))\
        
en_batch, zh_batch = next(iter(train_ds))
print("英文索引序列的batch")
print(en_batch)
print("中文索引序列的batch")
print(zh_batch)


# s2:model:transformer
# s2 - p1 - scaled dot product attention: QKV + softmax * V
