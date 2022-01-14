# -*- coding: utf-8 -*-
'''
* 生态：
tensorflow serving
tensorflow-lite
tensorflow in javascript
docker
...


* 部署准备：
1.模型保存
    tf.saved_model.save
    文件结构：
        * .pd图型结构
        * variables保存训练权重
        * assets添加外部文件
2.模型查看
    tf.keras.Model 保存时候自动生成signature_def
    saved_model_cli命令行窗口：
        * saved_model_cli show [-h] --dir Dir [--all]命令行查看详细信息
        * saved_model_cli run --dir Dir 运行模型，三种--input指定输入

        
* 使用docker + tensorflow serving部署
    * docker
        docker ps:查看当前容器
        docker images：查看已有镜像
    * tf serving：便于部署到生产环境
'''