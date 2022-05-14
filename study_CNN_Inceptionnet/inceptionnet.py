# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 15:37
# @Author  : Qisx
# @File    : inceptionnet.py
# @Software: PyCharm
import os.path
import time

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense


class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.model(inputs,training=False)
        return x


class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
        self.c4_1 = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, inputs, training=None, mask=None):
        x1 = self.c1(inputs)
        x2_1 = self.c2_1(inputs)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(inputs)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.c4_1(inputs)
        x4_2 = self.c4_2(x4_1)

        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x


class InceptionNet10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(InceptionNet10, self).__init__()
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.init_ch = init_ch
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        self.c1 = ConvBNRelu(init_ch, kernelsz=3)

        self.blocks = Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            self.out_channels = self.out_channels * 2

        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = InceptionNet10(num_blocks=2, num_classes=10)

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    checkpoint_save_path = 'checkpoint/InceptionNet10.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    cp_callback = ModelCheckpoint(filepath=checkpoint_save_path,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  monitor='val_loss')

    history=model.fit(x_train, y_train,
              batch_size=64,
              epochs=5,
              validation_data=(x_test, y_test),
              validation_freq=1,
              callbacks=[cp_callback])

    model.summary()

    file = open('checkpoint/weights.txt', 'a+')
    file.write(time.ctime())
    file.write('------------*-----------------------------*--------------' + '\n')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write((str(v.numpy())) + '\n')
        file.write('\n')
    file.write('\n')
    file.write('---------------------------------------------------------' + '\n')

# ######################## show ####################### #
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

if __name__ == '__main__':
    main()
