# -*- coding: utf-8 -*-
'''
tensorboard:训练过程与模型结构的可视化工具
fit/自定义训练过程均可以使用

tf.keras.callbacks.Tensorboard(参数...)

scalars界面：训练过程loss,训练速度,学习率等
graphs：可视化模型
distributions histograms：张量随时间分布，可视化权重和偏差

使用:
* fit训练：直接写入callbacks参数列表即可
tensorboard_callback = tf.keras.callbacks.Tensorboard(参数...);放在fit里
cmd：tensorboard --logdir path查看

* 自定义训练：tf.summary
语句：
tf.summary.create_file_writer(参数...)

tf.summary.image()
tf.summary.scalar()
tf.summary.text()
tf.summary.histogram()
tf.summary.audio()

查看graph和profile信息
tf.summary.trace_export()
tf.summary.trace_off()
tf.summary.trace_on()

实现过程：注意这时如果在model-call中定义静态图@tf.function会报错，因为train、test
import os
timestamp = datatime.datatime.now().strftime("%Y%m%d-%H%M%S)
logdir = os.path.join("log/"+timestamp)

summary_writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)

for epoch in range(EPOCHS):
    ...
    with summary_writer.as_default():
        跟踪语句

with summary_writer.as_default():
    tf.summary.trace_export(name='model_trace',step...)

cmd：tensorboard --logdir path查看
'''