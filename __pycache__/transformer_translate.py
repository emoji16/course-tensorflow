# -*- coding: utf-8 -*-
'''
transformer:兼顾上下文+并行

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
'''