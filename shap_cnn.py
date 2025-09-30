import os
import time

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import lime
from lime import lime_tabular
import numpy as np
import shap

datapath="dataset/BrainMethod.csv"
dataset_filename = datapath.split("/")[-1]
dataset_name = os.path.splitext(dataset_filename)[0]
model_save_dir = "shap_model"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, f"{dataset_name}_model.keras")

df = pd.read_csv(datapath, encoding="gbk")
x, y = df.iloc[:, 1:-1], df.iloc[:, -1]
from imblearn.over_sampling import SVMSMOTE
sm = SVMSMOTE(random_state=42)
x, y = sm.fit_resample(x, y)
# columns_to_drop = shap_DB_cnn_result
# df_dropped = x.drop(columns=columns_to_drop)
# x = df_dropped
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train = x_train.values.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1],1))
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train.values.reshape(len(y_train), 1))
y_test = onehot.fit_transform(y_test.values.reshape(len(y_test), 1))
print(f'训练集的形状为：{x_train.shape}')

inp=Input(shape=(x_train.shape[1:]))
x = Conv1D(32, 6, padding="same", activation='tanh')(inp)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(64, kernel_size=6, strides=1, padding='same', activation='relu')(x)
x = Conv1D(64, 6, strides=1, padding='same', activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(128, kernel_size=6, strides=1, padding='same', activation='relu')(x)
x = Conv1D(128, 6, strides=1, padding='same', activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(256, kernel_size=6, strides=1, padding='same', activation='relu')(x)
x = Conv1D(256, 6, strides=1, padding='same', activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Flatten()(x)
output = Dense(len(y.unique()), kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='softmax')(x)
model = Model(inputs=inp,outputs=output)

model.compile(optimizer = tf.keras.optimizers.Adam(0.001),  #优化器
              loss = 'categorical_crossentropy', #损失函数
              metrics = ['accuracy']
             )
lr_reduce=keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.5,min_lr=0.000001)
start = time.perf_counter()
history = model.fit(x_train,
                    y_train,
                    epochs=30,
                    callbacks = [lr_reduce],
                    batch_size= 32,
                    validation_data = (x_test,y_test),
                    shuffle=True)
end = time.perf_counter()

model.save(model_save_path)
print(model.evaluate(x_test,y_test))
print('Running time: %s Seconds'%(end-start))
Y_test = np.argmax(y_test, axis=1)
from sklearn.metrics import classification_report
predict = model.predict(x_test)
y_pred=np.argmax(predict,axis=1)
print(classification_report(Y_test, y_pred,digits=5))

model = load_model(model_save_path)
df = pd.read_csv(datapath, encoding="gbk")
x, y = df.iloc[:, 1:-1], df.iloc[:, -1]
from imblearn.over_sampling import SVMSMOTE
sm = SVMSMOTE(random_state=42)
x, y = sm.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train = x_train.values.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1],1))

#SHAP解释器初始化
explainer = shap.GradientExplainer(model,x_train)

#SHAP值计算
start = time.perf_counter()
shap_values = explainer.shap_values(x_test)  ## 全量测试集会非常耗时
end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))

#维度压缩
shap_values_new = shap_values[0].squeeze(axis=-1)  # 多分类任务处理方式错误
x_test_new = x_test.squeeze(axis=-1)
column_names = df.columns[1:-1].tolist()

#可视化
shap.summary_plot(shap_values_new,x_test_new,plot_type='bar',feature_names=column_names)

shap_test_mul_feature = np.abs(shap_values_new).mean(axis=0)
total = sum(shap_test_mul_feature)
shap_DB_cnn = [(x / total) * 100 for x in shap_test_mul_feature]

df = pd.read_csv(datapath, encoding="gbk")
column_names = df.columns[1:-1].tolist()
shap_DB_cnn_result = []
for i, name in enumerate(column_names):
    if shap_DB_cnn[i] < 0.2:
        shap_DB_cnn_result.append(name)
print(len(shap_DB_cnn_result))

df = pd.read_csv(datapath, encoding="gbk")
x, y = df.iloc[:, 1:-1], df.iloc[:, -1]
from imblearn.over_sampling import SVMSMOTE
sm = SVMSMOTE(random_state=42)
x, y = sm.fit_resample(x, y)
columns_to_drop = shap_DB_cnn_result
df_dropped = x.drop(columns=columns_to_drop)
x = df_dropped
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train = x_train.values.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1],1))
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train.values.reshape(len(y_train), 1))
y_test = onehot.fit_transform(y_test.values.reshape(len(y_test), 1))
print(f'训练集的形状为：{x_train.shape}')

inp=Input(shape=(x_train.shape[1:]))
x = Conv1D(32, 6, padding="same", activation='tanh')(inp)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(64, kernel_size=6, strides=1, padding='same', activation='relu')(x)
x = Conv1D(64, 6, strides=1, padding='same', activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(128, kernel_size=6, strides=1, padding='same', activation='relu')(x)
x = Conv1D(128, 6, strides=1, padding='same', activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(256, kernel_size=6, strides=1, padding='same', activation='relu')(x)
x = Conv1D(256, 6, strides=1, padding='same', activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Flatten()(x)
output = Dense(len(y.unique()), kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='softmax')(x)
model = Model(inputs=inp,outputs=output)

model.compile(optimizer = tf.keras.optimizers.Adam(0.001),  #优化器
              loss = 'categorical_crossentropy', #损失函数
              metrics = ['accuracy']
             )
lr_reduce=keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.5,min_lr=0.000001)
start = time.perf_counter()
history = model.fit(x_train,
                    y_train,
                    epochs=30,
                    callbacks = [lr_reduce],
                    batch_size= 32,
                    validation_data = (x_test,y_test),
                    shuffle=True)
end = time.perf_counter()

print(model.evaluate(x_test,y_test))
print('Running time: %s Seconds'%(end-start))
Y_test = np.argmax(y_test, axis=1)
from sklearn.metrics import classification_report
predict = model.predict(x_test)
y_pred=np.argmax(predict,axis=1)
print(classification_report(Y_test, y_pred,digits=5))