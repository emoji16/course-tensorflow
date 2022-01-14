# -*- coding: utf-8 -*-
'''
TF.hub : 复用预训练模型/迁移学习
pip install tensorflow-hub
hub.KerasLayer(handle,...)

bert实践：相似问题匹配问题
input: 向量token embed + 段向量 segment embed + 位置向量 position embed
input -> transformer-encoder * 12/24 -> + 具体应用layer

pre-training + fine-tuning 下游任务：天然支持下游任务，不只是提供embed
文本分类，语句匹配，问答系统，序列标注(分词BIE,答案抽取...)

# p1 - 分析数据：
# * 长度：10-15words 
# * 分布：问题类别/label/test-val比例
# p2 - 模型流程
# input - BERT(embedding - encoder - 交互) - 全连接 -输出
# 对于tf.hub中bert ,input:tokenization分割id序列, input_mask,seg_id序列(pos内置)
# 1 - import bert.tokenization:word-piece切割，截断过长句子
# 构建数据集，保存在tfrecords中：
# 2 - 合并sentence1 sentence2
# 3 - 不全id序列,构建input_mask,seg_id序列
'''
