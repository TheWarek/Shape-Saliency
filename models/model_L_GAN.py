from __future__ import print_function, division

from .model import ModelGAN
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPool2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import math

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
    return K.mean(K.square((y_pred - y_true)* m) - K.square((y_true - y_pred) * m), axis=-1)

def binary_crossentropy(y_true, y_pred):
    # return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    # return K.mean(K.square(y_pred - y_true), axis=-1)
    t_loss = K.max(y_pred, 0) - y_pred * y_true + K.log(1 + K.exp((-1) * K.abs(y_pred)))
    return K.mean(t_loss)

class L_GAN(ModelGAN):
    def __init__(self, w, h, c, batch_size=32, lr=0.001):
        super(L_GAN, self).__init__(w, h, c, batch_size)

        optimizer = Adam(lr=lr, beta_1=0.5)

        custom_loss = self.gauss_mse_loss(41, 400)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', #'binary_crossentropy', #binary_crossentropy mean_squared_error
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=custom_loss, optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.img_rows, self.img_cols, self.channels,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        m = Input(shape=(self.img_rows, self.img_cols, 1,))

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
                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                         input_shape=noise_shape))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

        # 2.
        model.add(Conv2D(filters=24, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

        # 3.
        model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

        # 4.
        model.add(Conv2D(filters=48, kernel_size=3, strides=1, padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(MaxPool2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

        # Deconvolution

        # 5.
        model.add(Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2,2)))

        # 6.
        model.add(Conv2D(filters=24, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2, 2)))

        # 7.
        model.add(Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2, 2)))

        # 8. -> final saliency map
        model.add(Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='same', data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('sigmoid'))
        #model.add(LeakyReLU(alpha=0.2))

        #
        #model.add(Dense(np.prod(self.sal_shape), activation='sigmoid'))
        #model.add(Reshape(self.sal_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        # input shape as saliency - fake/real ones?
        model.add(Conv2D(8, kernel_size=3, strides=2, input_shape=self.sal_shape, padding="same",
                         data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint=None, bias_constraint=None))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint=None, bias_constraint=None))
        #model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(Conv2D(24, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(Conv2D(48, kernel_size=3, strides=2, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=1, strides=1, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(Conv2D(16, kernel_size=1, strides=1, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(Conv2D(8, kernel_size=1, strides=1, padding="same", data_format='channels_last',
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint=None, bias_constraint=None))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
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

        img = Input(shape=self.sal_shape) #img_shape
        validity = model(img)

        return Model(img, validity)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.img_rows, self.img_cols, self.channels))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/results/sal_%d.png" % epoch)
        plt.close()

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
        # train_gt_glob = f['train_gt_glob'][()]
        # valid_gt_glob = f['valid_gt_glob'][()]

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

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
            noise = train_loc[idx]
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)