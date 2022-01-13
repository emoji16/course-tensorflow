# -*- coding: utf-8 -*-
'''
S9-Practice: 利用word2vec预训练词向量 + lstm 进行新闻文本分类

1. 文本表示方法
基于one-hot/ bag-of-words：tf-idf , textRank -- 稀疏，未考虑多义、同义性
topic model：LSA, pLSA, LDA -- 引入topic
基于词向量的固定表征：word2vec,fastText,glove
基于词向量的动态表征：elmo,GPT,BERT

2. word2vec:
* CBOW 多对一: 速度更快，对更频繁的单词有更好表示
* skip-gram 一对多：可以很好处理少量数据，可以很好表示稀疏单词
* 训练方法-tricks:
    * 层次softmax：对低频词效果好
    * sigmoid+负采样(每次不用更新所有w) ：向量维度低时效果好，对高频词效果好
    ps.负采样的本质-每次让一个训练样本只更新部分权重，其他权重全部固定；减少计算量
    区分：负采样 与 下采样(针对高频词)
* usage：gensim库 + jieba分词 可以增量训练

3. 实践 使用LSTM进行新闻文本分类
nlp-intro：nlu,nlg,application
s0-分析数据
s1-构建数据集
* 文本处理：分词，构建词典，转换为序列形式
* 文本截断(前 后)，文本补全（前 后）:
*   截断：tf.keras.preprocessing.text.Tokenizer(num_words,filters,lower,split,char_level,oov_token)
            tf.keras.preprocessing.sequence.pad_sequences
*   后补全: tf.data.Dataset.padded_batch(batch_size,padded_shapes)
* one-hot编码label
s2-词向量构建+word embedding矩阵构建
s3-构建model
    * 采用rnn(多层双向lstm) 
    * 一维池化 (attention池化，平均池化，最大池化)
    * 全连接层
        tricks：
        * dropout：tf.keras.layers.Dropout(rate,noise_shape,seed)
        * batch normalization：按批normalize + 线性转换，提高训练速度，加快收敛，增强分类效果
            tf.keras.layers.BatchNormalization()
    * softmax多分类
s4-评估优化
'''
# 源数据分析
# 文本长度
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from pandas import DataFrame as df

# df = pd.read_csv('data_news/train.tsv',header=None,delimiter='\t',names=['label','content'])
# df['content_len'] = [len(article) for article in df['content']]
# len_mean = np.mean(df['content_len'].tolist())
# len_80 = np.percentile(df['content_len'].values,80)
# len_85 = np.percentile(df['content_len'].values,85)
# print(len_mean, len_80, len_85)
# plt.figure(figsize=(20,10))
# plt.plot(df['content_len'].tolist(), marker='*',markerfacecolor='red')
# plt.axhline(y=len_mean,color='black')
# plt.axhline(y=len_80,color='peru')
# plt.axhline(y=len_85,color='orange')
# plt.show()

# 源数据分析-分布统计
# count_class = df['label'].value_counts()
# plt.figure(figsize=(20,10))
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# class_bar = plt.bar(x=count_class.index, height = count_class.tolist(),width=0.4)
# plt.xticks()
# plt.yticks()
# for bar in class_bar:
#     height = bar.get_height()
#     plt.text(bar.get_x()+bar.get_width() /2,height+1,str(height),ha='center',va='bottom')
# plt.xlabel('类别')
# plt.ylabel('sample count')
# plt.show()

# PART1 -build dataset,预训练embedding库构建
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
train = pd.read_csv('./data_news/train.tsv', sep='\t', header=None, names=['label','content'])
val = pd.read_csv('./data_news/dev.tsv', sep='\t', header=None, names=['label','content'])
test = pd.read_csv('./data_news/test.tsv', sep='\t', header=None, names=['label','content'])
# print(train['content'][0])
# print(train.shape)
# print(train.head())
train = train.iloc[:100,:]
val = val.iloc[:10,:]
test = test.iloc[:20,:]

# s1-分词
import jieba
def content_cut(x):
    x = jieba.lcut(x)
    x = ' '.join(x)
    return x

train['content'] = train['content'].map(lambda x: content_cut(x))
val['content'] = val['content'].map(lambda x: content_cut(x))
test['content'] = test['content'].map(lambda x: content_cut(x))

# s2-样本分析
df =pd.concat([train,val,test], axis=0)
# df['content_len'] = df['content'].map(lambda x: len(x.split(' ')))
# print(np.percentile(df['content_len'].values, 80))  # 

# s2-训练word-embedding model(用于预训练 word_embedding)
# 对整个语料库document建立word-embedding model
import os 
model_file_name = './embedding/Word2Vec_word_200.model'
sentences = [document.split(' ') for document in df['content'].values]
if not os.path.exists(model_file_name):
    model = Word2Vec(sentences = sentences,vector_size=200,window=5,epochs=10,workers=11,seed=2022,min_count=2)
    model.save(model_file_name)
else:
    model = Word2Vec.load(model_file_name)
print("word2vec built.")

# s3-转换成词索引序列
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000,lower=False,filters="")
tokenizer.fit_on_texts(df['content'].tolist())
train_ = tokenizer.texts_to_sequences(train['content'].values)
val_ = tokenizer.texts_to_sequences(val['content'].values)
test_ = tokenizer.texts_to_sequences(test['content'].values)
# print(train_[0])

# s4-样本补全与截断
train_ = tf.keras.preprocessing.sequence.pad_sequences(train_,maxlen=800)
val_ = tf.keras.preprocessing.sequence.pad_sequences(val_,maxlen=800)
test_ = tf.keras.preprocessing.sequence.pad_sequences(test_,maxlen=800)
# tf.keras.preprocessing.sequence.pad_sequences

# s5- 保存索引-预训练embedding
word_vocab = tokenizer.word_index
count = 0
embedding_matrix = np.zeros((len(word_vocab)+1,200))
for word, i in word_vocab.items():
        embedding_vector = model.wv[word] if word in model.wv else None
        if embedding_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_vector
        else :
            unk_vec = np.random.random(200)*0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec
print(embedding_matrix.shape)

# s6-label encoder + onehot
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
lb = LabelEncoder()
train_label = lb.fit_transform(train['label'].values)
train_label = to_categorical(train_label,num_classes=10,dtype='int') # onehot
val_label = lb.transform(val['label'].values)
val_label = to_categorical(val_label,num_classes=10,dtype='int') # onehot
test_label = lb.transform(test['label'].values)
test_label = to_categorical(test_label,num_classes=10,dtype='int') # onehot

# PART2 -build model
content = tf.keras.layers.Input(shape=(800),dtype='int32')  # batch * 800 * 200

# s1-embedding layer构建 + 预训练用法
# embedding-way1-预训练
embedding = tf.keras.layers.Embedding(name='word_embedding',
        input_dim=embedding_matrix.shape[0],
        weights=[embedding_matrix],
        output_dim = embedding_matrix.shape[1],
        trainable=False)
# embedding-way2-embedding layer-不写weights, trainble = False

# s2-dropout-SpatialDropOut1D
x = tf.keras.layers.SpatialDropout1D(0.2)(embedding(content))

# s3-gru-bi-rnn 双层
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(200,return_sequences=True))(x)  #  将得到的正反向拼接起来 output (batch,800,400)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(200,return_sequences=True))(x)

# s4-avg_pool + max_pool 的操作 (concatenate)--针对sequence 800words进行pooling
avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)  # 直接降维至 batch* 400
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
conc = tf.keras.layers.concatenate([avg_pool,max_pool]) # batch * 800

# s4-dense + batchnormalization
x = tf.keras.layers.Dense(1000)(conc) 
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(500)(x) 
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Dense(10)(x) 
# s5-softmax
output = tf.nn.softmax(x)

model = tf.keras.models.Model(inputs=content, outputs=output)

# PART 3: train
train_ds = tf.data.Dataset.from_tensor_slices((train_,train_label))
train_ds = train_ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size = 23000)
train_ds = train_ds.batch(batch_size=32)

val_ds = tf.data.Dataset.from_tensor_slices((val_,val_label))
val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.shuffle(buffer_size = 23000)
val_ds = val_ds.batch(batch_size=32)

for a,b in train_ds.take(1):
    print(a.shape, b.shape)

lr = 0.001
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_acc')

def train_one_step(contents, labels):
    with tf.GradientTape() as tape:
        predictions = model(contents)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

def val_one_step(contents, labels):
    predictions = model(contents)
    t_loss=loss_object(labels, predictions)

    val_loss(t_loss)
    val_accuracy(labels, predictions)

EPOCHS = 1
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

    for content,labels in train_ds:
        train_one_step(content, labels)
    
    for val_content,val_labels in val_ds:
        val_one_step(val_content,val_labels)

    template = 'EPOCH: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy:{}'
    print(template.format(epoch+1, 
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        val_loss.result(),
                        val_accuracy.result()*100
        ))

# 测试集表现
test_ds = tf.data.Dataset.from_tensor_slices(test_)
test_ds = test_ds.batch(batch_size=32)
predictions=[]
for line in test_ds:
    prediction = model(line)
    predictions.extend(list(np.argmax(prediction.numpy(),axis=1)))

from sklearn.metrics import accuracy_score,classification_report
test_true = list(np.argmax(test_label,axis=1))
print(accuracy_score(test_true, predictions))

print(classification_report(test_true, predictions,target_names=list(lb.classes_)))
    

