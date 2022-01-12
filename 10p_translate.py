# -*- coding: utf-8 -*-
'''
实践：使用transformer进行机器翻译
    s2s过程-teaching fools策略：input + label-targetlanguage[:,:-1] -> 生成-targetlanguage[:,1:] 
    注意前后加上了BOF-EOF
    * s1-数据准备
        * src-data: https://data.statmt.org/news-commentary/v14/
        * 还是用tf.keras.preprocessing进行切割和字典的构建
            tf.keras.preprocessing.text.Tokenizer
            tf.keras.preprocessing.sequence.pad_sequences
        * 每个句子序列化(jieba+nltk)+前后加上开始、结束token
        * 过滤长文本train_ds.filter.batch.shuffle.prefetch
    * s2-手写transformer model
    PART0-定义scaled dot product attention方法;封装mha-layer,ffn-layer';positional encoding计算;
        * 定义attention k,q,v,mask计算func：scaled dot product attention
            * 注意留multi-head维
            * 注意mask两种用法
                * padding mask:识别实际内容，盖住补0
                * look ahead mask：decoder中不偷窥后面
                * 对于decoderLayer第一个attention 要做到两者兼顾combined_mask 
        * 定义multi-head attention类-sublayer(定义一个大layer类：QKV+ dense layer类实现)
        * 定义ffn func-sublayer(d_model,dff稍大)
        * 定义positional encoding function
        * 定义mask：padding mask / look-forward mask function
    PART1-encoder部分{embedding,[encoderLayer{mha{qkv,dense}, ffn}]}
        * 组装encoderLayer类:mha + ffn (dropout + layernorm + resnet)
        * 组装encoder类： embedding encoderLayer 
    PART2-decoder部分{embedding,[decoderLayer{mha-self{qkv,dense},mha{qkv,dense}, ffn}]}
        * 组装decoderLayer类：mha + encoder-decoder attention + ffn
        * 组装decoder类： embedding decoderLayers
    PART3-Transformer-model组装{encoder,decoder,linearLayer}
    * s3 - demo test
    * s4 - train
'''

# key pnt：
# Q1-solved: 序列长度如何通知:input(batch_size,seq_len)
# Q2-solved: mask维度 batch*1*1*seq_len(0位置0)
# 原因：broadcasting
# import tensorflow as tf
# a=tf.constant([[[1,1],[1,1]],[[1,1],[1,1]]]) # 2*2*2
# print(tf.shape(a))
# b = tf.constant([[-1,-1],[-1,-1]]) # 2,1,2
# print(tf.shape(b))
# print(a+b)
# print(tf.shape(a+b))
# Q3-solved：q每次传入？batch_size针对样本一个q--对每一个单词创建三个向量：word_embed * 训练矩阵QKV得到
# 训练的是Dense:wq,wk,wv,共享参数
# Q4-solved：output1 进入wq(q) ; en_output传递进入wk(k),wv(v) ，Q'K'V'还是得训练
# Q5-solved: input_vocab_len, target_vocab_len均指定,指定的是序列值的区间，以供embedding使用
#            qkv的参数由 embedding-x 决定，这里都用了序列长度seq_len
# Q6-solved: look_ahead_mask 直接按照tar_seq_size对角矩阵即可完成

import os
import io
from re import I
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.ops.control_flow_v2_toggles import output_all_intermediates
from tensorflow.python.ops.gen_array_ops import InplaceSub
from tensorflow.python.ops.gen_experimental_dataset_ops import experimental_assert_next_dataset
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

en_vocab_size = 10000
zh_vocab_size = 10000

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
        
# en_batch, zh_batch = next(iter(train_ds))
# print("英文索引序列的batch")
# print(en_batch)
# print("中文索引序列的batch")
# print(zh_batch)

# s2:model:transformer ()
# s2 - p1 - scaled dot product attention: QKV + softmax * V
def scaled_dot_product_attention(q,k,v,mask):
    '''
    args: seq_len_k == seq_len_v
    q : shape == (...,seq_len_q, depth)  seq_len_q :即查询,长度对应一个句子序列
    k : shape == (...,seq_len_k, depth)
    v : shape == (...,seq_len_v, depth_v)
    mask: (...,1,seq_len_q)  # 作用于qk'

    return:
    attention_weights:(...,seq_len_q, seq_len_k) 理解：返回序列中每一个子词对序列k其他词的权重
    output:(...,seq_len_q, depth) 理解：返回序列中每一个子词的新表示
    '''
    matmul_qk = tf.matmul(q,k,transpose_b = True)  # (...,seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32) # dk：seq_k序列的长度depth
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # 因为要softmax,所以极小已经接近0了

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1) # (...,seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights,v)  # (...,seq_len_q, depth)
    
    return output, attention_weights

# s2 - p2 - 搭建multi-head attention类-sublayer(定义一个大layer类+多个dense QKV layer类实现)
    # d_model = num_heads * depth
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads= num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)  # q,k,v是根据每一个x(word)input * d_model矩阵计算出来的出来的
        self.wk = tf.keras.layers.Dense(d_model)  # 参数共享，wq,wk,wv
        self.wv = tf.keras.layers.Dense(d_model)  # 所以q,k,v 尺寸batch_size
        
        self.dense = tf.keras.layers.Dense(d_model) # 针对multi-head返回的multi-V进行一层加权

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0,2,1,3])  # (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q) # (...,seq_len_q,d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (..., num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (..., num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (..., num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q,k,v,mask
        )

        # scaled_attention : shape ==  (batch_size, num_heads, seq_len_q, depth)
        # attention_weights : shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1 , self.d_model)) # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)

        return output, attention_weights # (batch_size, seq_len_q, d_model), (batch_size, num_heads, seq_len_q, seq_len_k)

# s2 - p3 - feed forward networks-sublayer
# dff 常大于 d_model提取有用信息,论文中d_model ==512, dff == 2048
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([  # 不止model可用sequential
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

# s2 - p4 - 组装encoderLayer类-(mha+ffn+dropout+layer normalization+resnet)
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff ,rate = 0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # 2 sublayers:mha ffn，每个子层后面还有dropout 和 layernorm+resnet
        # sublayer1: mha
        # 这里实现过程seq_len_k == seq_len_q, depth == depth_v
        attn_output, attn =self.mha(x,x,x,mask)
        attn_output = self.dropout1(attn_output,training = training)  # 非trainning可以不dropout
        out1 = self.layernorm1(x + attn_output)  # resnet实现方式

        # sublayer1: ffn
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, seq_len_q, d_model)
        return out2

# s2 - p5 -decoderLayer类
# sublayers: self-attention + encoder-decoder-attention + ffn
# self-attention sublayer: QKV 都是自己+ 需要look ahead mask
# encoder-decoder attention sublayer：使用self-attention层输出序列作为Q k,v 使用encoder的输出序列
# 理解：参考已经生成的(中文)字词为当前output产生一个包含前文语义的Q,与encoder中的(英文)序列匹配，
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate = 0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, combined_mask, inp_padding_mask):
        # sublayer1:
        attn1, attn_weights_block1 = self.mha1 (x,x,x,combined_mask)
        attn1 = self.dropout1(attn1, training= training)
        out1 = self.layernorm1(attn1 + x)

        # sublayer2:
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, inp_padding_mask
        )
        attn2 = self.dropout2(attn2, training = training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training = training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

# s2 - p6 - positional encoding :二维sin/con 表达位置信息; 与原线段进行相加
# pe_fb:奇数sin,偶数cos
def get_angles(pos, i ,d_model):  # 位置pos，index_in_vector=i的值
    angle_rates = 1/np.power(10000, (2*(i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],  # 转置成为矩阵
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    sines = np.sin(angle_rads[:,0::2])
    cosines = np.cos(angle_rads[:,1::2])
    pos_encoding = np.concatenate([sines,cosines],axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype = tf.float32) # shape == (seq_len , d_model)

# s2- p7 - 组装encoder类：embedding + encoder_layers
# input_vocab_size 固定序列长度： fixed seq_len
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        input_seq_len = tf.shape(x)[1]  # (batch, tar_seq_len, d_model)

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # 原文缩放参数
        x += self.pos_encoding[:, :input_seq_len,:] # 注意：相加关系
        x = self.dropout(x,training = training)

        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x, training, mask)
            print('-'*20)
            print("EncoderLayer {i+1} output:", x)

        return x

# s2- p7 - 组装decoder类：embedding(label) + decoder_layers
# note: 如何利用output输入mha,见p8
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff ,target_vocab_size,rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training , combined_mask, inp_padding_mask): # 这是inp训练的inp_padding_mask
        tar_seq_len = tf.shape(x)[1] # (batch, tar_seq_len, d_model)
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tar_seq_len,:]
        x = self.dropout(x, training = training)

        for i , dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, training, combined_mask,inp_padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights
    
# s2- p8 - 组装transformer类：encoder+decoder+linear layer
# note: 不是对output进行attention,而是对label数据进行
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)  # 输出和中文字典大小向量，softmax进行翻译
    
    def call(self, inp ,tar, training, enc_padding_mask, combined_mask, dec_padding_mask):
        enc_output = self.encoder(inp,training,enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, combined_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights

# s2 - p9 - mask实现: padding mask + look ahead mask
# note：利用broadcast
# padding mask : 索引序列中0的补位设为1
def create_padding_mask(seq): # seq (batch,seq_len,d_model)
    mask = tf.cast(tf.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis,:]  # broadcasting

        # mask: (...,seq_len_q, seq_len_k,d_model)  # 作用于qk'
# look ahead mask : 索引序列中0的补位设为1

def create_look_ahead_mask(size):  # 上三角矩阵(target已有的size)
    mask = 1 - tf.linalg.band_part(tf.ones((size,size)),-1,0)  # 1-主对角线+整个下三角
    return mask

# combined_mask = tf.maximum(tar_padding_mask, look_ahead_mask)

# s3 - demo - test
demo_examples = [
    ("In any case, let’s wish him luck.", "无论如何，让我们祝他好运。"),
    ("The stakes for Africa are enormous.", "这对非洲厉害攸关。")
]
# print(demo_example)

demo_ds = tf.data.Dataset.from_tensor_slices((
    [en for en, _ in demo_examples],[zh for _, zh in demo_examples]
))
demo_ds = demo_ds.map(tf_encode).padded_batch(batch_size=2, padded_shapes=([-1],[-1]))
inp,tar = next(iter(demo_ds))
# print('inp: ',inp) # (2,13)
# print('tar: ',tar) # (2,8)

vocab_size_en = en_vocab_size + 2 + 1
vocab_size_zh = zh_vocab_size + 2 + 1
d_model = 4
embedding_layer_en = tf.keras.layers.Embedding(vocab_size_en, d_model)
embedding_layer_zh = tf.keras.layers.Embedding(vocab_size_zh, d_model)
emb_inp = embedding_layer_en(inp) # (2,13,4)
emb_tar = embedding_layer_zh(tar) # (2,8,4)
# print(emb_inp)
# print(emb_tar)

inp_padding_mask = create_padding_mask(inp) # (2,1,1,13)
mask = tf.squeeze(inp_padding_mask, axis=1) # (2, 1, 13)
q = emb_inp # (2,13,4)
k = emb_inp # (2,13,4)
v = emb_inp # (2,13,4)
_, attention_weights = scaled_dot_product_attention(q,k,v,mask)

seq_len = emb_tar.shape[1] # 2*8*8
look_ahead_mask = create_look_ahead_mask(seq_len) # (2, 8, 8)
# print(look_ahead_mask)
# q = emb_tar # (2,8,4)
# k = emb_tar # (2,8,4)
# v = emb_tar # (2,8,4)

# _, attention_weights = scaled_dot_product_attention(q,k,v,look_ahead_mask)
# print(attention_weights) # 2,13,13 / 2,8,8

num_heads = 2
dff = 8

# mha = MultiHeadAttention(d_model, num_heads)
# output, attention_weights = mha(v,k,q,inp_mask)
# print(output.shape) # 2* 13 * 4
# print(attention_weights.shape) # 2* 2 * 13 * 13

enc_layer = EncoderLayer(d_model, num_heads, dff)
enc_out = enc_layer(emb_inp, training = False, mask = inp_padding_mask)
# print(enc_out.shape) # 2* 13 * 4

tar_padding_mask = create_padding_mask(tar)
look_ahead_mask = create_look_ahead_mask(tar.shape[-1])
combined_mask = tf.maximum(tar_padding_mask, look_ahead_mask)

dec_layer = DecoderLayer(d_model, num_heads, dff)
dec_out,_,_ = dec_layer(emb_tar, enc_out, False, combined_mask, inp_padding_mask)
# print(dec_out.shape) # 2* 8 * 4

# pos_encoding = positional_encoding(tf.constant(50),tf.constant(512))
# plt.pcolormesh(pos_encoding[0],cmap='RdBu')
# plt.xlabel('d_model')
# plt.xlim((0,512))
# plt.xlabel('Position')
# plt.colorbar()
# plt.show()

num_layers = 2
encoder = Encoder(num_layers, d_model, num_heads, dff, vocab_size_en)
enc_out = encoder(inp, training = False, mask=inp_padding_mask)
# print(enc_out.shape)
rate = 0.1
decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size_zh,rate)
dec_out = decoder(tar, enc_out, training = False, combined_mask=combined_mask, inp_padding_mask=inp_padding_mask)


tar_inp = tar[:,:-1]
tar_real = tar[:,1:]
tar_padding_mask = create_padding_mask(tar_inp)
look_ahead_mask = create_look_ahead_mask(tar_inp.shape[1])
combined_mask = tf.math.maximum(tar_padding_mask,look_ahead_mask)

transformer = Transformer(num_layers, d_model, num_heads , dff,vocab_size_en,vocab_size_zh)
predictions, _ = transformer(inp, tar_inp, False, inp_padding_mask, combined_mask, inp_padding_mask)
print(tar_real.shape)
print(predictions.shape)

# s4 - train