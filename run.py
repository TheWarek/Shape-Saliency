
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import random

import inference as inference
from tensorflow.examples.tutorials.mnist import input_data
import h5py
import cv2

import sys, getopt

def main(argv):
   stacks = 4
   dataset = 'mnist'
   loc = True
   glob = False
   noise = False
   try:
      opts, args = getopt.getopt(argv,"hLGn")
   except getopt.GetoptError:
      print('run.py -L (add local context) -G (add global context) -n (to include noise in batch)')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('run.py -L (add local context) -G (add global context) -n (to include noise in batch)')
         sys.exit()
      elif opt == '-L':
          loc = True
      elif opt == '-G':
          glob = True
      elif opt == '-n':
          noise = True
   #run_train(loc, glob, noise)

if __name__ == "__main__":
   main(sys.argv[1:])