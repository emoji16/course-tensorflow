# -*- coding: utf-8 -*-
'''
tf.data
* tf.data.Dataset
    * p1-创建数据集:其中元素迭代 
        ds = tf.data.Dataset.from_tensors()  # 可读元组/字典,不切割，返回单元素数据集
        ds = tf.data.Dataset.from_tensor_slices() # 可读元组/字典，会把input的第0维看做dataset size
        ds = tf.data.Dataset.from_generator()  # 从preprocessing.dataGenerator中接收
        示例读取from_generator()：
        定义gen + 迭代输出的函数Gen() + 绑定ds
        img_gem = tf.keras.preprocessing.image.ImageDataGenerator(...)
        def Gen():
            gen = img_gen.flow_from_directory(filepath)
            for (x,y) in gen:
                yield(x,y)
        ds = tf.data.Dataset.from_generator(Gen可调用对象,output_types=,output_shapes=)

        使用：
        可以定义class DataLoader(object):
            def __init__():
            def __call__():
                def _generator():
                    pass 
                ds = tf.data.Dataset.from_generator(_generator,...)
                return ds
    * p2-transform:结合tf.io,tf.image,tf.stack..
        ds.map(f)
        ds.shuffle(buffer_size)
        ds.repeat(cnt)
        ds.batch(batch_size)
        ds.flat_map(f) # 将map中每个元素中list各item单独作为元素
        ds.interleave(f, cycle_length, block_lenth,) 
        # 作用：并行数据加载，将不同csv文件结合起来
        # 针对整个数据集，取出cycle_length个元素，对每个元素分别apply function得到cycle_length个新dataset
        # 轮流从结果数据集中取一次取block_lenth，cycle
        ds.cache()
        ds.skip(n)
        ds.take(n) 
        ds.zip()  # 横向铰合
        ds.concatenate()  # 纵向连接
        ds.reduce()
        ds.filter()

        # 性能优化部分，构建高效数据管道：
        # prefetch(并行data和train)，interleave(多进程)，cache(第一个epoch后存于内存，省open，read，适用于小数据)，map(num_parallel_calls多进程执行)
        # 多进程参数num_parallel_calls = tf.data.experimental.AUTOTUNE

* tf.data.TFRecordDataset  
    TFRecord：tensorflow高效处理的数据集存储格式 .tfrecords
    tf.data.TFRecordDataset(filenames, compression_type=None,buffer_size=None,num_parallel_reads=None)
    
    写入tfrecords步骤：
        * s1-由内存中input数据建立tf.train.Feature字典
            value是list有以下三种类型，input必须是list类型 [image]
            * bytes_list:string bytes
                'key1':tf.train.Feature(bytes_list=tf.train.BytesList(value=intput1))
            * float_list:float32, float64
            * int64_list:bool enum int32 uint32 int64 uint64
        * s2-将数据元素转换为tf.train.Example对象，注意：tf.train.Features
            每一个tf.train.Example由若干个tf.train.Feature字典（key-value)组成
        * s3-序列化+写入tf.train.Example元素组成的列表文件，使用tf.io.TFRecordWriter()写入TFRecord文件

        * 示例写入tfrecords：
        with tf.io.TFRecordWriter(test_tfrecord_file) as writer:
            for filename,label in zip(test_filenames, test_labels):

                # s1-定义features
                image = open(filename,'rb).read()  # 读取图片到内存
                feature= {
                    'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])), # 注意只有一个image也要是list！
                    'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
                
                # s2-构建example
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # s3-序列化+写入文件TFRecords
                # serialized = example.SerializeToString()  
                writer.write(serialized)

    读取tfrecords步骤：
        * s1-tf.data.TFRecordDataset读入原始TFRecord文件，获得tf.data.Dataset
            note:tf.train.Example对象还是序列化的
        * s2-定义Feature结构
        * s3-通过dataset的ds.map方法，对每一个序列化tf.train.Example字符串执行tf.io.parse_single_example函数
            反序列化
            可封装为_parse_example函数
        * 示例读取tfrecords：
            * s2 - 定义feature结构:feature_description
                feature_description = {
                    'image': tf.io.FixedLenFeature([],tf.string),
                    'label': tf.io.FixedLenFeature([],tf.int64)
                }
            * s3 - 定义解码函数：_parse_example 用tf.io.parse_io_single_example 将example_stirng和feature_description结合
                # 将TFRecord文件中每一个序列化example解码
                def _parse_example(example_string):
                    feature_dict = tf.io.parse_io_single_example(example_string, feature_description)
                    # 进一步具体处理
                    feature_dict['image] = tf.io.decode_jpeg(feature_dict['image'])
                    feature_dict['image] = tf.image.resize(feature_dict['image'],[256,256]) / 255.0
                    return feature_dict['image'], feature_dict['label]
            * s1 - 读取文件，对整个数据集批量解码 等操作
                train_ds = tf.data.TFRecordDataset("train.tfrecords")
                train_dataset = train_ds.map(_parse_example)
                # shuffle batch prefetch...

* tf.data.TextLineDataset
    行为元素 tf.string
    tf.data.TextLineDataset(filenames, compression_type=None,buffer_size=None,num_parallel_reads=None)

    示例读取line：
    def data_func(line):
        line = tf.strings.split(line, sep = ',')
        return line
    
    lines = tf.data.TextLineDataset()
    line_ds = lines.skip(1).map(data_func) # 丢弃第一行表头

''' 
