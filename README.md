1. basic concepts：
   * low level APIs: tensor & basic operations
   * mid level APIs: layers-related: tf.keras.layers API + tf.keras.layers.Layer子类进行自定义
   * high level APIs: model-related: tf.keras.Model子类进行自定义

2. model part：whole process
   
   how to build graph: sesssion, eager execution, autograph(subclass + @tf.function修饰call / @tf.function修饰train_step、test_step)
   
   * build ： 3 ways（sequential model， functional model， subclass model）
   * compile(loss, metrics, optimizer) ：2 ways (model.compile, GradientTape+train)
   * fit: 2 ways (model.fit, GradientTape+train+batch+epoch，可以按此分为keras模型与自定义模型)
   * evaluate: 2 ways (model.compile+model.evaluate, tf.keras.metrics)
   * predict: 2 ways (model.predict , subclass-model(x))
   * save/load model: 
     * model.save_weights/model.load_weights : weights only /checkpoints-- 需要model部分代码才能predict
     
       * model.save_weights("adasd.h5") *# .h5格式*
     * model.save_weights('./checkpoints/mannul_checkpoint') # checkpoint格式
       
     * model.save/tf.keras.models.load_model: weights(if after fit), architecture, optimizer configuration # HDF5格式
   
       load得到model可以直接predict
     
       * ( filepath, 可选save_format='tf'为pb)
       * (filename ,.h5)
     
     * tf.saved_model.save/load: weights, architecture。常用于部署
     
       由于load结果不是一个model,还要经过f = restored_saved_model.signatures["serving_default"]调用指定函数f进行predict(注意input要转换为tensor tf.constant)
     
       cmd: saved_model_cli show --dir ./data/ --all
     
       *  (model, filepath)
   * others
     * tf.keras.utils.plot_model
     * `print`(model.summary())
   
3. layers：

   * tf.keras.layers  api

   * 自定义layer
     * class MyLayer ： `__init__`, build, call, get_config(字典形式传入init中变量)

       三种初始化参数方式:tf.random_normal_initializer 、self.add_weight()、build(inputshape):self.add_weight

     * model.save需要变量初始化时指定名字w/b

     * tf.keras.models.load_model 指定custom_objects

4. loss:

   * tf.keras.losses api
   * 自定义loss
     * 类实现：继承 tf.keras.losses.Loss：`__init__`,call
     * 函数式实现：def f1 def f2 (y_true, y_pred) return loss return f2

5. metrics：
   * tf.keras.metrics api
   * 自定义loss
     - 类实现：__init__,update_state,reset_state,result
     - 函数式实现

6. tensorboard

   * keras.model.fit：训练过程写在callbacks参数列表里

     自定义训练中：需要利用tf.summary

   * cmd：tensorboard --logdir path查看

7. data:tf.data

   * tf.data.DataSet
   * tf.data.TFRecordDataset
     * 写入tfrecords文件：定义tf.train.Feature, 定义tf.train.Example,examples.SerializeToString()序列化+tf.io.TFRecordWriter写入
     * 读取tfrecords文件：定义features_description,利用tf.io.parse_io_single_example解码，使用map批量解码
   * tf.data.TextLineDataset
   
8. cnn

9. rnn + project: word2vec_lstm 新闻分类实践

10. transformer 

11. 分布式训练tf.distributed.Strategy:gpu tpu 

12. tf.hub + project: 获取预训练bert 相似问题判断

13. deployment : docker + tf.serving. 三种测试-curl，requests.post,+flask中间件



  
