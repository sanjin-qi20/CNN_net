# -*- coding: utf-8 -*-
# @Time    : 2022/5/5 19:35
# @Author  : Qisx
# @File    : CNN_baseline.py
# @Software: PyCharm
import os.path
import time

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

class BaseLine(Model):
    def __init__(self):
        super(BaseLine, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='SAME')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        self.d1 = Dropout(0.2)

        self.flatten = Flatten()
        self.Da1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.22)
        self.Da2 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.Da1(x)
        x = self.d2(x)
        y = self.Da2(x)
        return y


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 225.0, x_test / 225.0

model = BaseLine()
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = 'BaseLine.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('*******************load the model***********************')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                              save_weights_only=True,
                              save_best_only=True,
                              )
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=5,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

file = open('weights.txt', 'a+')
file.write(time.ctime() + '\n')
file.write('******************************************' + '\n')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
    file.write(str(history.history)+'\n')
    file.write('\n')
file.write('\n\n')
file.close()

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
