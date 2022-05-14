# -*- coding: utf-8 -*-
# @Time    : 2022/5/11 10:40
# @Author  : Qisx
# @File    : CNN_resnet.py
# @Software: PyCharm
import os.path
import time

from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, MaxPool2D


class ResnetBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides,padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=filters, strides=1, kernel_size=(3, 3), padding='same', use_bias=False)
        self.b2 = BatchNormalization()
        if residual_path:
            self.down_c1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()
        self.a2 = Activation('relu')

    def call(self, inputs, training=None, mask=None):
        residual = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)
        return out


class ResNet18(Model):
    def __init__(self, block_list, initial_filters=64):
        super(ResNet18, self).__init__()
        self.block_list = block_list
        self.block_num = len(block_list)
        self.out_filters = initial_filters

        self.c1 = Conv2D(filters=self.out_filters,strides=1, kernel_size=(3, 3), padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = Sequential()

        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, strides=1, residual_path=False)
                self.blocks.add(block)
            self.out_filters = self.out_filters * 2

        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(10, activation='softmax', kernel_regularizer=l2())

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = ResNet18([2, 2, 2, 2])

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = 'checkpoint/Resnet.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('----------------------------load the model---------------------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = ModelCheckpoint(filepath=checkpoint_save_path,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  monitor='val_loss'
                                  )

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=5,
                        validation_data=(x_test, y_test),
                        validation_freq=1,
                        callbacks=[cp_callback]
                        )
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
