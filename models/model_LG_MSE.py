from __future__ import print_function, division

from .model import ModelGAN
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPool2D, Softmax
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Concatenate
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import TensorBoard
import math
from random import randint

import matplotlib.pyplot as plt
import h5py

import sys

import numpy as np
from utils.util import matlab_style_gauss2D

#m = K.variable(matlab_style_gauss2D((400,400),41))

def gauss_crossentropy(y_true, y_pred, sigma, kernel):
    m = K.variable(matlab_style_gauss2D((kernel, kernel), sigma))
    t_loss = K.max(y_pred, 0) - y_pred * y_true + K.log(1 + K.exp((-1) * K.abs(y_pred)))
    t_loss = t_loss * m
    return K.mean(t_loss)

def gauss_mse(y_true, y_pred, sigma, kernel):
    m = K.variable(matlab_style_gauss2D((kernel, kernel), sigma))
    a = K.mean(K.square((y_pred - y_true)), axis=-1)
    return K.mean(a * m, axis=-1)
    # return K.mean((K.square((y_pred - y_true)) - K.square((y_true - y_pred))) * m, axis=-1)

def binary_crossentropy(y_true, y_pred):
    # return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    # return K.mean(K.square(y_pred - y_true), axis=-1)
    t_loss = K.max(y_pred, 0) - y_pred * y_true + K.log(1 + K.exp((-1) * K.abs(y_pred)))
    return K.mean(t_loss)

def mse_custom(y_true, y_pred):
    return K.mean(K.square((y_pred - y_true)), axis=-1)

class LG_MSE(ModelGAN):
    def __init__(self, w, h, c, glob_c, batch_size=32, lr=0.0001):
        super(LG_MSE, self).__init__(w, h, c, glob_c, batch_size)

        optimizer = Adam(lr=lr, beta_1=0.5)

        custom_loss = self.gauss_mse_loss(0.5, 400)

        self.reg = None #regularizers.l2(0.00001)
        self.loss = 'mean_squared_error'

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator

        self.generator = self.build_generator()
        self.generator.compile(optimizer=optimizer, loss=self.loss )

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.img_rows, self.img_cols, self.channels,))
        z_true = Input(shape=(self.img_rows, self.img_cols, 1,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator([img, z_true])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([z, z_true], valid)
        # self.combined = Model(z, [valid, z_true])
        # self.combined = Model(z, valid)
        self.combined.compile(loss=self.loss, optimizer=optimizer)

        # ---- GLOBAL ----

        # Build and compile the discriminator
        self.glob_discriminator = self.build_glob_discriminator()
        self.glob_discriminator.compile(loss=self.loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        # The generator takes noise as input and generated imgs
        glob_z = Input(shape=(self.img_rows, self.img_cols, self.glob_channels,))

        self.glob_generator = self.build_glob_generator()
        self.glob_generator.compile(optimizer=optimizer, loss=self.loss)

        self.generator.trainable = False
        self.discriminator.trainable = False
        self.glob_discriminator.trainable = False

        glob_img = self.generator(z)
        glob_img2 = self.glob_generator([glob_z, glob_img])
        glob_valid = self.glob_discriminator([glob_img2, z_true])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.glob_combined = Model([z, glob_z, z_true], glob_valid)
        self.glob_combined.compile(loss=self.loss, optimizer=optimizer)


    def gauss_crossentropy_loss(self, sigma, kernel):
        def custom_loss(y_true, y_pred):
            return gauss_crossentropy(y_true, y_pred, sigma, kernel)
        return custom_loss

    def gauss_mse_loss(self, sigma, kernel):
        def custom_loss(y_true, y_pred):
            return gauss_mse(y_true, y_pred, sigma, kernel)
        return custom_loss

    def build_generator(self):
        noise_shape = self.img_shape

        model = Sequential()

        # 1.
        model.add(Conv2D(filters=16, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None,
                         input_shape=noise_shape))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # 2.
        model.add(Conv2D(filters=24, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # 3.
        model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # 4.
        model.add(Conv2D(filters=48, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # Deconvolution

        # 5.
        model.add(Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2,2)))
        model.add(Dropout(0.25))

        # 6.
        model.add(Conv2D(filters=24, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Dropout(0.25))

        # 7.
        model.add(Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Dropout(0.25))

        # 8. -> final saliency map
        model.add(Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('sigmoid'))
        #model.add(LeakyReLU(alpha=0.2))

        #
        #model.add(Dense(np.prod(self.sal_shape), activation='sigmoid'))
        #model.add(Reshape(self.sal_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        #added = Input(shape=self.sal_shape)
        img = model(noise)

        #img = Concatenate(axis=-1)([img,noise[:,:,:,:]])

        #model.summary()
        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        # regularizers.l2(0.01)
        # input shape as saliency - fake/real ones?
        model.add(Conv2D(8, kernel_size=3, strides=2, input_shape=self.sal_shape, padding="same",
                         data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        # model.add(Conv2D(8, kernel_size=3, strides=2, input_shape=self.sal_shape_merged, padding="same",
        #                  data_format='channels_last',
        #                  dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        #                  bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        #                  activity_regularizer=regularizers.l2(0.01), kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        #model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(24, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(48, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=1, strides=1, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(16, kernel_size=1, strides=1, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(8, kernel_size=1, strides=1, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # model.add(Dense(4096))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        #model.add(Activation('sigmoid'))

        model.summary()

        img = Input(shape=self.sal_shape)
        img2 = Input(shape=self.sal_shape)
        #img3 = Concatenate(axis=-1)([img, img2])
        #validity = model(img3)
        validity = model(img)

        return Model([img, img2], validity)

    def build_glob_generator(self):
        noise_shape = self.img_glob_shape_merged

        model = Sequential()

        # 1.
        model.add(Conv2D(filters=36, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None,
                         input_shape=noise_shape))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # 2.
        model.add(Conv2D(filters=48, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # 3.
        model.add(Conv2D(filters=60, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # 4.
        model.add(Conv2D(filters=72, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # Deconvolution

        # 5.
        model.add(Conv2D(filters=60, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2,2)))
        model.add(Dropout(0.25))

        # 6.
        model.add(Conv2D(filters=36, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Dropout(0.25))

        # 7.
        model.add(Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Dropout(0.25))

        # 8. -> final saliency map
        model.add(Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('sigmoid'))

        #model.add(LeakyReLU(alpha=0.2))

        #
        #model.add(Dense(np.prod(self.sal_shape), activation='sigmoid'))
        #model.add(Reshape(self.sal_shape))

        model.summary()

        noise = Input(shape=self.img_glob_shape)
        loc_gen = Input(shape=self.sal_shape)
        img_merge = Concatenate(axis=-1)([noise, loc_gen])
        img = model(img_merge)

        return Model([noise, loc_gen], img)

    def build_glob_discriminator(self):

        model = Sequential()
        # input shape as saliency - fake/real ones?
        # model.add(Conv2D(8, kernel_size=3, strides=2, input_shape=self.sal_shape_merged, padding="same",
        #                  data_format='channels_last',
        #                  dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        #                  bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        #                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        model.add(Conv2D(8, kernel_size=3, strides=2, input_shape=self.sal_shape, padding="same",
                         data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        #model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(24, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(48, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=1, strides=1, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(16, kernel_size=1, strides=1, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(8, kernel_size=1, strides=1, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=self.reg, kernel_constraint=None, bias_constraint=None))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # model.add(Dense(4096))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.sal_shape)
        img2 = Input(shape=self.sal_shape)
        # img3 = Concatenate(axis=-1)([img, img2])
        # validity = model(img3)
        validity = model(img)

        return Model([img, img2], validity)

    def save_imgs(self, valid_loc, valid_glob, valid_gt_loc, valid_gt_glob, epoch, loc=True, train=False):
        r, c = 5, 5
        #noise = np.random.normal(0, 1, (r * c, self.img_rows, self.img_cols, self.channels))
        noise = valid_loc[0:r*c]
        noise_gt = valid_gt_loc[0:r*c]

        glob_noise = valid_glob[0:r * c]
        glob_noise_gt = valid_gt_glob[0:r * c]

        if loc:
            gen_imgs = self.generator.predict(noise)
        else:
            gen_imgs = self.glob_generator.predict([glob_noise, self.generator.predict(noise)])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        if loc:
            if train:
                fig.savefig("images/results/train/local/sal_%d.png" % epoch)
            else:
                fig.savefig("images/results/local/sal_%d.png" % epoch)
        else:
            if train:
                fig.savefig("images/results/train/global/sal_%d.png" % epoch)
            else:
                fig.savefig("images/results/global/sal_%d.png" % epoch)
        plt.close()

        if epoch == 0:
            # ground truth
            fig2, axs2 = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    if loc:
                        axs2[i, j].imshow(noise_gt[cnt, :, :, 0], cmap='gray')
                    else:
                        axs2[i, j].imshow(glob_noise_gt[cnt, :, :, 0], cmap='gray')
                    axs2[i, j].axis('off')
                    cnt += 1

            if loc:
                if train:
                    fig2.savefig("images/results/train/local/0_sal_%d_gt.png" % epoch)
                else:
                    fig2.savefig("images/results/local/0_sal_%d_gt.png" % epoch)

            else:
                if train:
                    fig2.savefig("images/results/train/global/0_sal_%d_gt.png" % epoch)
                else:
                    fig2.savefig("images/results/global/0_sal_%d_gt.png" % epoch)

            plt.close()

    def write_log(self, callback, scope, names, logs, batch_no):
        with tf.name_scope(scope):
            for name, value in zip(names, logs):
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = scope + name
                callback.writer.add_summary(summary, batch_no)
                callback.writer.flush()

    def nss_score(self, saliency_map, ground_truth):
        # saliencyMap is the saliency map
        # fixationMap is the human fixation map(binary matrix)

        F = ground_truth.flatten()
        x = np.nonzero(F > 0.)

        if len(x[0]) == 0:
            return 0.0

        # normalize saliency map
        s_map = (saliency_map - np.mean(saliency_map)) / np.std(saliency_map)

        s_map = s_map.flatten()

        where = np.array(list(map(lambda x: 1 if x else 0, F)))
        where = np.nonzero(where > 0)

        score = np.mean(s_map[where])

        return score


    def auc_score(self, saliency_map, ground_truth, n_splits = 100, stepSize = 0.1):
        # normalize saliency Map
        saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

        #flatten
        S = saliency_map.flatten()
        F = ground_truth.flatten()

        # 0. is probably too low cause lowest is like 1e-2
        x = np.nonzero(F > 0.)
        if len(x[0]) == 0:
            return 0.0
        #sal map values at fixation locations
        Sth = S[x]
        n_fix = len(Sth)
        n_pix = len(S)

        # for each fixation, sample nsplits from anywhere on the sal map
        r = np.random.randint(1, n_pix, [n_fix, n_splits])
        # sal map values at random locations
        randfix = S[r]

        # calculate AUC per random split
        auc = np.zeros((n_splits))
        for s in range(0, n_splits):
            curfix = randfix[:, s]
            allthreshes = np.flipud(np.arange(0, np.amax([Sth, curfix]),stepSize))

            tp = np.zeros((len(allthreshes) + 2))
            fp = np.zeros((len(allthreshes) + 2))
            tp[0] = 0
            tp[-1] = 1
            fp[0] = 0
            fp[-1] = 1

            for i in range(0, len(allthreshes)):
                thresh = allthreshes[i]
                tp[i+1] = sum(j >= thresh for j in Sth) / n_fix
                fp[i+1] = sum(j >= thresh for j in curfix) / n_fix

            auc[s] = np.trapz(fp, tp)

        score = np.mean(auc)

        return score

    def train(self, epochs, batch_size=32, save_interval=50, dataset_file='./datasets/data.h5'):

        f = h5py.File(dataset_file, 'r')

        # # List all groups
        # print("Keys: %s" % f.keys())
        # a_group_key = list(f.keys())[0]

        # Get the data
        train_loc = f['train_loc'][()]
        valid_loc = f['valid_loc'][()]
        train_glob = f['train_glob'][()]
        valid_glob = f['valid_glob'][()]

        train_gt_loc = f['train_gt_loc'][()]
        valid_gt_loc = f['valid_gt_loc'][()]
        train_gt_glob = f['train_gt_glob'][()]
        valid_gt_glob = f['valid_gt_glob'][()]

        valid_binary_gt_loc = f['valid_binary_gt_loc'][()]
        train_binary_gt_loc = f['train_binary_gt_loc'][()]
        valid_binary_gt_glob = f['valid_binary_gt_glob'][()]
        train_binary_gt_glob = f['train_binary_gt_glob'][()]

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # For validation
        v_valid = np.ones((valid_gt_loc.shape[0], 1))
        v_fake = np.zeros((valid_gt_loc.shape[0], 1))

        # callback
        log_path = './logs'
        callback = TensorBoard(log_path)
        callback.set_model(self.combined)
        train_names = ['train_loss', 'train_accuracy']
        val_names = ['val_loss', 'val_accuracy']

        g_mse_loc_loss = 0
        g_mse_glob_loss = 0
        g_loss = 0

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, train_loc.shape[0], batch_size)
            imgs = train_gt_loc[idx]

            # Sample noise and generate a batch of new images
            # noise = np.random.normal(0, 1, (batch_size, self.img_rows, self.img_cols, self.channels))
            # Instead of noise we will have our models included
            # TODO EDIT: not noise, but the input
            noise = train_loc[idx]
            gen_imgs = self.generator.predict(noise) #imgs

            # Append basic shapes along predicted saliency to the discriminator

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch([imgs, noise[:,:,:,0:1]], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, noise[:,:,:,0:1]], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #test = self.discriminator.metrics_names

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([noise, noise[:,:,:,0:1]], valid)

            # Also train generator to avoid confusion of overall big model => also need less step to convergence
            g_mse_loc_loss = self.generator.train_on_batch(noise, imgs)

            # Plot the progress
            print("LOCAL MSE GEN: %d [G loss: %f]" % (epoch, g_mse_loc_loss))
            print("LOCAL: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                with tf.name_scope('LOCAL'):
                    # Discriminator

                    self.write_log(callback, 'LOCAL/DISC/', ['train_loss'], [d_loss[0]], epoch)

                    valid_gen_imgs = self.generator.predict(valid_loc)
                    valid_d_loss_real = self.discriminator.test_on_batch([valid_gt_loc, valid_loc[:, :, :, 0:1]], v_valid)
                    valid_d_loss_fake = self.discriminator.test_on_batch([valid_gen_imgs, valid_loc[:, :, :, 0:1]], v_fake)
                    valid_d_loss = 0.5 * np.add(valid_d_loss_real, valid_d_loss_fake)
                    # callback
                    self.write_log(callback, 'LOCAL/DISC/', ['valid_loss'], [valid_d_loss[0]], epoch)

                    # Generator
                    self.write_log(callback, 'LOCAL/GEN/', ['train_mse_loss'], [g_mse_loc_loss], epoch)
                    self.write_log(callback, 'LOCAL/GEN/', ['train_loss'], [g_loss], epoch)
                    g_mse_loc_loss = self.generator.test_on_batch(valid_loc, valid_gt_loc)
                    self.write_log(callback, 'LOCAL/GEN/', ['valid_mse_loss'], [g_mse_loc_loss], epoch)
                    g_loss = self.combined.test_on_batch([valid_loc, valid_loc[:, :, :, 0:1]], v_valid)
                    self.write_log(callback, 'LOCAL/GEN/', ['valid_loss'], [g_loss], epoch)

                self.save_imgs(valid_loc, valid_glob, valid_gt_loc, valid_gt_glob, epoch, loc=True)
                self.save_imgs(train_loc, train_glob, train_gt_loc, train_gt_glob, epoch, loc=True, train=True)

                # AUC and NSS score

                train_auc_score = 0
                train_nss_score = 0
                valid_auc_score = 0
                valid_nss_score = 0
                train_auc_del = 0
                train_nss_del = 0
                valid_auc_del = 0
                valid_nss_del = 0
                for i in range(0, gen_imgs.shape[0]):
                    auc_score = self.auc_score(gen_imgs[i, :, :, 0], train_binary_gt_loc[i, :, :, 0])
                    if auc_score == 0:
                        train_auc_del += 1
                    train_auc_score += auc_score
                    nss_score = self.nss_score(gen_imgs[i, :, :, 0], train_binary_gt_loc[i, :, :, 0])
                    if nss_score == 0:
                        train_nss_del += 1
                    train_nss_score += nss_score

                train_auc_score /= (gen_imgs.shape[0] - train_auc_del)
                train_nss_score /= (gen_imgs.shape[0] - train_nss_del)

                for i in range(0, valid_gen_imgs.shape[0]):
                    auc_score = self.auc_score(valid_gen_imgs[i, :, :, 0], valid_binary_gt_loc[i, :, :, 0])
                    if auc_score == 0:
                        valid_auc_del += 1
                    valid_auc_score += auc_score
                    nss_score = self.nss_score(valid_gen_imgs[i, :, :, 0], valid_binary_gt_loc[i, :, :, 0])
                    if nss_score == 0:
                        valid_nss_del += 1
                    valid_nss_score += nss_score

                valid_auc_score /= (valid_gen_imgs.shape[0] - valid_auc_del)
                valid_nss_score /= (valid_gen_imgs.shape[0] - valid_nss_del)

                self.write_log(callback, 'LOCAL/AUC/', ['train_auc'], [train_auc_score], epoch)
                self.write_log(callback, 'LOCAL/AUC/', ['valid_auc'], [valid_auc_score], epoch)
                self.write_log(callback, 'LOCAL/NSS/', ['train_nss'], [train_nss_score], epoch)
                self.write_log(callback, 'LOCAL/NSS/', ['valid_nss'], [valid_nss_score], epoch)


            # ---- GLOBAL ----

            # ---------------------
            #  Train Global Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, train_glob.shape[0], batch_size)
            loc_imgs = train_gt_loc[idx]
            glob_imgs = train_gt_glob[idx]

            # Sample noise(input) and generate a batch of new images
            loc_noise = train_loc[idx]
            glob_noise = train_glob[idx]

            gen_glob_imgs = self.glob_generator.predict([glob_noise, self.generator.predict(loc_noise)])  # imgs

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.glob_discriminator.train_on_batch([glob_imgs, glob_noise[:, :, :, 0:1]], valid)
            d_loss_fake = self.glob_discriminator.train_on_batch([gen_glob_imgs, glob_noise[:, :, :, 0:1]], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Global Generator
            # ---------------------


            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.glob_combined.train_on_batch([loc_noise, glob_noise, glob_noise[:, :, :, 0:1]], valid)

            # Also train generator to avoid confusion of overall big model => also need less step to convergence
            g_mse_glob_loss = self.glob_generator.train_on_batch([glob_noise, self.generator.predict(loc_noise)],
                                                                 glob_imgs)

            # Plot the progress
            print("GLOBAL MSE GEN: %d [G loss: %f]" % (epoch, g_mse_glob_loss))
            print("GLOBAL: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                with tf.name_scope('GLOBAL'):
                    # Discriminator

                    self.write_log(callback, 'GLOBAL/DISC/', ['train_loss'], [d_loss[0]], epoch)


                    valid_gen_glob_imgs = self.glob_generator.predict([valid_glob, self.generator.predict(valid_loc)])  # imgs
                    valid_d_loss_real = self.glob_discriminator.test_on_batch([valid_gt_glob, valid_glob[:, :, :, 0:1]], v_valid)
                    valid_d_loss_fake = self.glob_discriminator.test_on_batch([valid_gen_glob_imgs, valid_glob[:, :, :, 0:1]], v_fake)
                    valid_d_loss = 0.5 * np.add(valid_d_loss_real, valid_d_loss_fake)

                    self.write_log(callback, 'GLOBAL/DISC/', ['valid_loss'], [valid_d_loss[0]], epoch)

                    # Generator

                    self.write_log(callback, 'GLOBAL/GEN/', ['train_mse_loss'], [g_mse_glob_loss], epoch)
                    self.write_log(callback, 'GLOBAL/GEN/', ['train_loss'], [g_loss], epoch)

                    valid_g_mse_glob_loss = self.glob_generator.test_on_batch([valid_glob, self.generator.predict(valid_loc)], valid_gt_glob)
                    self.write_log(callback, 'GLOBAL/GEN/', ['valid_mse_loss'], [valid_g_mse_glob_loss], epoch)
                    g_loss = self.glob_combined.test_on_batch([valid_loc, valid_glob, valid_glob[:, :, :, 0:1]], v_valid)
                    self.write_log(callback, 'GLOBAL/GEN/', ['valid_loss'], [g_loss], epoch)

                self.save_imgs(valid_loc, valid_glob, valid_gt_loc, valid_gt_glob, epoch, loc=False)
                self.save_imgs(train_loc, train_glob, train_gt_loc, train_gt_glob, epoch, loc=False, train=True)

                # AUC and NSS score

                train_auc_score = 0
                train_nss_score = 0
                valid_auc_score = 0
                valid_nss_score = 0
                train_auc_del = 0
                train_nss_del = 0
                valid_auc_del = 0
                valid_nss_del = 0
                for i in range(0,gen_glob_imgs.shape[0]):
                    auc_score = self.auc_score(gen_glob_imgs[i, :, :, 0], train_binary_gt_glob[i, :, :, 0])
                    if auc_score == 0:
                        train_auc_del += 1
                    train_auc_score += auc_score
                    nss_score = self.nss_score(gen_glob_imgs[i, :, :, 0], train_binary_gt_glob[i, :, :, 0])
                    if nss_score == 0:
                        train_nss_del += 1
                    train_nss_score += nss_score

                train_auc_score /= (gen_glob_imgs.shape[0] - train_auc_del )
                train_nss_score /= (gen_glob_imgs.shape[0] - train_nss_del )

                for i in range(0, valid_gen_glob_imgs.shape[0]):
                    auc_score = self.auc_score(valid_gen_glob_imgs[i, :, :, 0], valid_binary_gt_glob[i, :, :, 0])
                    if auc_score == 0:
                        valid_auc_del += 1
                    valid_auc_score += auc_score
                    nss_score = self.nss_score(valid_gen_glob_imgs[i, :, :, 0], valid_binary_gt_glob[i, :, :, 0])
                    if nss_score == 0:
                        valid_nss_del += 1
                    valid_nss_score += nss_score

                valid_auc_score /= (valid_gen_glob_imgs.shape[0] - valid_auc_del)
                valid_nss_score /= (valid_gen_glob_imgs.shape[0] - valid_nss_del)

                self.write_log(callback, 'GLOBAL/AUC/', ['train_auc'], [train_auc_score], epoch)
                self.write_log(callback, 'GLOBAL/AUC/', ['valid_auc'], [valid_auc_score], epoch)
                self.write_log(callback, 'GLOBAL/NSS/', ['train_nss'], [train_nss_score], epoch)
                self.write_log(callback, 'GLOBAL/NSS/', ['valid_nss'], [valid_nss_score], epoch)
