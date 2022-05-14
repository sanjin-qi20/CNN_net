# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 11:37
# @Author  : Qisx
# @File    : CNN_alexnet.py
# @Software: PyCharm
import os
import time

from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, Dropout


class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3), padding='valid')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3), padding='Valid')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same')
        self.a3 = Activation('relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same')
        self.a4 = Activation('relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.a5 = Activation('relu')
        self.p5 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.a3(x)

        x = self.c4(x)
        x = self.a4(x)

        x = self.c5(x)
        x = self.a5(x)
        x = self.p5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y


def plot_loss_acc(history):
    acc = history['sparse_categorical_accuracy']
    val_acc = history['val_sparse_categorical_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

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


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = AlexNet()
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = 'checkpoint/AlexNet.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('*-----------------------load the model--]--------------------------*')
        model.load_weights(checkpoint_save_path)
    cp_callback = ModelCheckpoint(filepath=checkpoint_save_path,
                                  monitor='val_loss',
                                  save_best_only=True,
                                  save_weights_only=True)
    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=5,
                        validation_data=(x_test, y_test),
                        validation_freq=1,
                        callbacks=[cp_callback])
    model.summary()
    file = open('checkpoint/weights.txt', 'a')
    file.write(time.ctime())
    file.write('\n')
    file.write('--------------------------------------------------------------------------------')
    file.write('\n')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
        file.write(str(history.history) + '\n')
    file.write('\n\n')

    plot_loss_acc(history.history)


if __name__ == '__main__':
    main()
