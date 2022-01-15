# -*- coding: utf-8 -*-
'''

docker + tf.serving部署model到服务器上
docker run / docker stop / 
测试：curl /requests.post / flask使用

* 生态：
tensorflow serving
tensorflow-lite
tensorflow in javascript
docker

* 部署准备：
1.模型保存
    tf.saved_model.save
    文件结构：
        * .pd图型结构
        * variables保存训练权重
        * assets添加外部文件
2.saved_model_cli命令行查看模型细节，验证是否保存成功
    tf.keras.Model 保存时候自动生成signature_def
    saved_model_cli命令行窗口：(P1-word2vec_lstm.py最后实践)
        * saved_model_cli show [-h] --dir Dir [--all]命令行查看详细信息 [--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
        * saved_model_cli run --dir Dir [--inputs INPUTS] [--input_exprs INPUTS_EXPRS] [--input_examples ]
        
* 使用docker + tensorflow serving部署
    * 安装docker以及常见指令
        sudo apt install docker.io
        sudo docker ps:查看当前容器container
        sudo docker images：查看已有镜像
    * 下载tensorflow serving仓库, 查看测试样例
        * git clone https://github.com/tensorflow/serving  
        * cd ../serving/tensorflow_serving/servables/tensorflow/testdata
        *cd saved_model_half_plus_two_cpu
    * 在docker上拉取tensorflow serving镜像
        * docker pull tensorflow/serving  # 获取tensorflow docker镜像
    * 在docker上投放镜像，运行模型
        * docker run ：
            * 绑定原始端口 -p /路径到期望路径 -v  ,注意路径中不写版本号
            开放gPRC 8500/REST API 8501端口，如将rest api8501发送到主机8501
            * -e:my_model填充环境变量MY_MODEL, MODEL_BASE_PATH使用默认
            * 使用tensorflow/serving镜像
        TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"
        docker run -t --rm -p 8501:8501
        -v $TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two
        -e MODEL_NAME=half_plus_two
        tensorflow/serving &
    * 测试：
        * curl进行测试,收到结果
            curl -d '{"instances": [1.0, 2.0, 3.0]}' -X POST http://114.116.50.39:8501/v1/models/half_plus_two:predict
        * requests.post + json 测试
            // 一开始ping不通，安全组-入方向规则-全部放通 -- fixed
            a = [1.0, 2.0, 3.0]
            data = json.dumps({"signature_name":"serving_default", "instance":[a]})
            headers = {"content-type": "application/json"}
            json_response = requests.post('http://114.116.50.39:8501/v1/models/half_plus_two:predict', data = data, headers=headers)
            predictions = json.loads(json_response.text)['predictions']
        * 中间件flask：requests - 中间件(可以补充中间数据处理) - url
            from flask import flask
    * docker stop
        sudo docker stop container_id

'''



# 部署测试1：服务器端
# curl -d '{"instances": [1.0, 2.0, 3.0]}' -X POST http://114.116.50.39:8501/v1/models/half_plus_two:predict

# 部署测试2：requests.post
import json, requests

requests.adapters.DEFAULT_RETRIES = 5
a = [1.0, 2.0, 3.0]
data = json.dumps({"signature_name":"serving_default", "instances":[a]})
headers = {"content-type": "application/json","Connection":"close"}
json_response = requests.post('http://114.116.50.39:8501/v1/models/half_plus_two:predict', data = data, headers=headers)
predictions = json.loads(json_response.text)
print(predictions)

# 部署测试3：+ flask 中间件做数据处理
from flask import Flask, request
import pickle

app = Flask(__name__)

