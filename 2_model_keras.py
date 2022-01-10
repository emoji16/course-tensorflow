# -*- coding: utf-8 -*-
'''
a whole process of build a model by functional method+ compile+fit+evaluate+predict
train model in keras:tf.keras.Model
build+compile
train-fit:callbacks-tf.keras.callbacks(ModelCheckpoint定期保存模型, EarlyStopping, TensorBoard, CSVLogger
                                      ,ReduceLROnPlateau动态学习率调整：通过回调访问验证指标的变化优化学习率) 
tf.keras.utils.plot_model 可以进行模型可视化
bug坑：虚拟环境配置问题 在prompt里install graphviz和pydot就好了
evaluate(test dataset)
predict
'''

import tensorflow as tf
print(tf.__version__)

# build 
def get_uncompiled_model():
    inputs = tf.keras.Input(shape=(32,),name='digits')
    x = tf.keras.layers.Dense(64, activation='relu',name='dense_1')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu',name='dense_2')(x)
    outputs = tf.keras.layers.Dense(10,name='predictions')(x)
    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
    # 如果是多output也可以定义一个/多个loss function==[]或者{'':},可以补充loss_weight{'':}联合训练
    # metrics也可以多个[[],]
    
    return model
        
# train 
import numpy as np

x_train = np.random.random((1000,32))
y_train = np.random.randint(10,size=(1000,))

x_val = np.random.random((200,32))
y_val = np.random.randint(10,size=(200,))

x_test = np.random.random((200,32))
y_test = np.random.randint(10,size=(200,))

# fit过程可以有sample_weight和class_weight
# class_weight:affects the relative weight of each class in the calculation of the objective function
# sample_weight:解决样本质量不同的问题，为每个样本进行加权。例如时序数据某时间段可信度更高
class_weight = {0:1, 1:1, 2:1, 3:1, 4:1, 5:2, 6:1, 7:1, 8:1, 9:1}
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2
model = get_compiled_model()
# help(tf.keras.utils.plot_model)
tf.keras.utils.plot_model(model,'model_structure.png',show_shapes=True,dpi=500)
# fit 回调函数callbacks可以利用参数列表传入
# EarlyStopping: monitor, min_delta, patience, verbose, mode(min,max,auto)
# ModelCheckpoint:filepath, save_best_only, monitor, save_weights_only, verbose
# ReduceLROnPlateau:monitor,verbose,mode,factor学习率衰减因子,patience,min,cooldown,min_lr
# help(tf.keras.callbacks.EarlyStopping)
callbacks = [
    # tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',  # 验证集上的loss
    #     min_delta=1e-2,
    #     patience=2,
    #     verbose=1
    # ),
    # tf.keras.callbacks.ModelCheckpoint(
    #     filepath = "MyModel_{epoch}",
    #     save_best_only = True,
    #     monitor = 'val_loss',
    #     save_weights_only = True,
    #     verbose = 1,
    # )，
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        verbose = 1,
        factor=0.5,
        patience=3
    )
]

# help(model.fit)
# 多输入输出时可以{'':}、[],()传入
model.fit(x_train, y_train, 
        class_weight = class_weight,
        sample_weight = sample_weight,
        callbacks = callbacks,
        batch_size=64, 
        epochs = 30,
        validation_data=(x_val,y_val),
        steps_per_epoch=1) # -使用val数据集
# model.fit(x_train, y_train, batch_size=64, validation_split=0.2,epochs=1,steps_per_epoch=1)

# evaluate-使用test测试集
print("\n # Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size = 128)
print("test loss, test_acc", results)

# predict
print("\n # Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions :", predictions)
