# -*- coding: utf-8 -*-
'''
概念：worker device..

Part1-GPU
分布式训练distributed strategy:
并行方式:模型并行(各node输入相同数据，运行模型的不同部分;通信开销大) vs 数据并行(不同node输入不同数据，运行相同的完整模型；更常用)
数据并行的参数更新算法：parameter server模式(单设ps, worker) vs all reduce模式(ps, worker角色统一，所有node参数拷贝一致--实现改进有ring-allreduce等) 
    * ps拿到参数之后又可分为：同步更新 vs 异步更新
        * 同步更新（所有worker发来梯度和参数请求后取平均更新；更常用）
        * 异步更新：各个worker异步计算、迭代--无锁迭代可能被丢掉，需要小学习率--浪费数据和资源
    * ring-allreduce:5张卡环形相连，每张卡有左手卡和右手卡，一个负责接收，一个负责发送：循环4次完成梯度积累，再完成4次参数同步

api:
    * tf.distribute.MirroredStrategy -- 同一台机器多个GPU上同步分布式训练(每一个gpu上有一个相同副本，且更新同步) + all-reduce在设备间传递更新
        tf.distribute.MirroredStrategy(devices=[])
    * tf.distribute.MirroredStrategy(devices=[])
    * tf.distribute.experimental.MultiWorkerMirroredStrategy()
    * tf.distribute.experimental.CentralStorageStrategy()  # 所有参数存于cpu
    * tf.distribute.experimental.ParameterServerStrategy()

使用：
    * model: 创建tf.distribute.Strategy + model， model.compile放在  .scope()下即可
    * 自定义方式-ds:需要创建分布式数据集：分发 dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    * loss_fn: loss(reduction=false),单独写函数 用tf.nn.compute_average_loss求和+总体平均(global_barch_size)
      单独写的话loss 和 metrics也要写在.scope()下面
    * 自定义方式-train: mirrored_strategy.run(train_step) + mirrored_strategy.reduce
    train_step, val_step,mirrored_strategy.run也都要写在.scope()下面
    keras model- fit不用

gpu内存管理：gpus = tf.config.experimental.list_physical_devices('GPU')
        * 限制内存
          tf.config.experimental.set_memory_growth
          tf.config.experimental.set_visible_devices(gpu[0],'GPU')
        * 虚拟GPU
          tf.config.experimental.set_virtual_device_configuration(gpu[0],'GPU')
          logical_gpus = tf.config.experimental.list_logical_devices
        * 虚拟多个CPU
          tf.config.experimental.set_virtual_device_configuration(gpu[0],[
              tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
          ])
          logical_gpus
    
Part2-TPU
'''