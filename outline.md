1. basic concepts：
   * low level APIs: tensor & basic operations
   * mid level APIs: layers-related: tf.keras.layers API + tf.keras.layers API.Layer子类进行自定义
   * high level APIs: model-related: tf.keras.Model子类进行自定义

3. whole process
   * build ： 3 ways（sequential model， functional model， subclass model）
   * compile(loss, metrics, optimizer) ：2 ways (model.compile, GradientTape+train)
   * fit: 2 ways (model.fit, GradientTape+train+batch+epoch，可以按此分为keras模型与自定义模型)
   * evaluate: 2 ways (model.compile+model.evaluate, tf.keras.metrics)
   * predict: 2 ways (model.predict , subclass-model(x))
   * save/load model: 
     * model.save_weights : weights only /checkpoints-- 需要model部分代码才能predict
     
       load得到model可以直接predict
     
       * model.save_weights("adasd.h5") *# .h5格式*
       * model.save_weights('./checkpoints/mannul_checkpoint') # checkpoint格式
    
     * model.save: weights(if after fit), architecture, optimizer configuration # HDF5格式
     
       load得到model可以直接predict
     
       * ( filepath, 可选save_format='tf'为pb)
       * (filename ,.h5)
     
     * tf.saved_model.save: weights, architecture。常用语于部署
     
       由于load结果不是一个model,还要经过f = restored_saved_model.signatures["serving_default"]调用指定函数f进行predict(注意input要转换为tensor tf.constant)
     
       cmd: saved_model_cli show --dir ./data/ --all
     
       *  (model, filepath)
   * others
     * tf.keras.utils.plot_model
     * print(model.summary())
   
4. how to build graph: sesssion, eager execution, autograph(subclass + @tf.function修饰call)