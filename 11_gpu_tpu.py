# -*- coding: utf-8 -*-
'''
分布式训练： distributed strategy
并行方式:模型并行(各node输入相同数据，运行模型的不同部分;通信开销大) vs 数据并行(不同node输入不同数据，运行相同的完整模型；更常用)
数据并行的参数更新算法：parameter server模式(单设ps, worker) vs all reduce模式(ps, worker角色统一，所有node参数拷贝一致--实现改进有ring-allreduce等) 
    * ps拿到参数之后又可分为：同步更新 vs 异步更新
        * 同步更新（所有worker发来梯度和参数请求后取平均更新；更常用）
        * 异步更新：各个worker异步计算、迭代--无锁迭代可能被丢掉，需要小学习率--浪费数据和资源
'''