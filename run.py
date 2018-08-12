
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.model_L_GAN import L_GAN

import numpy as np
import os
import random
import pprint
import tensorflow.contrib.slim as slim

from inference import SALSH
from tensorflow.examples.tutorials.mnist import input_data
import h5py
import cv2

import sys, getopt


def main(_):
   l_gan = L_GAN(w=400, h=400, c=7)
   l_gan.train(epochs=5000, batch_size=16, save_interval=50, dataset_file='./datasets/data.h5')
   return 0

if __name__ == '__main__':
   tf.app.run()