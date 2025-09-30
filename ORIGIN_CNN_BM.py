import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.layers import Dense,Flatten,Conv1D,MaxPooling1D,Dropout,BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from imblearn.over_sampling import SMOTE
df = pd.read_csv("data/BrainMethod.csv",encoding="gbk")
x, y = df.iloc[:, 1:-1], df.iloc[:, -1]

from imblearn.over_sampling import SVMSMOTE
sm = SVMSMOTE(random_state=42)
x, y = sm.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train = x_train.values.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1],1))
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train.values.reshape(len(y_train), 1))
y_test = onehot.fit_transform(y_test.values.reshape(len(y_test), 1))

# # CNN
inp=Input(shape=(x_train.shape[1:]))

x = Conv1D(32, 6, padding = "same", activation='tanh')(inp)
x = BatchNormalization()(x)
x = Conv1D(64, kernel_size=6, strides=1, padding='same',activation='relu')(x)#padding可能会导致靠近边界的部分相比于中间部分对于输出的的影响更小
                                                                              #可能会导致边界存在一定程度的欠表示
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(128, kernel_size=6, strides=1, padding='same',activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(256, kernel_size=6, strides=1, padding='same',activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
g = Flatten()(x)
output = Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation='softmax')(g)

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
Y_test = np.argmax(y_test, axis=1)# Convert one-hot to index
from sklearn.metrics import classification_report
predict = model.predict(x_test)
y_pred=np.argmax(predict,axis=1)
print(classification_report(Y_test, y_pred,digits=5))