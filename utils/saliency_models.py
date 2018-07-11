from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import numpy as np

import sys, getopt, os
from scipy.spatial import distance
import random



## INFO: This script will compute multiple local/global based saliency models for each shape/contour
# TODO: translate to Python