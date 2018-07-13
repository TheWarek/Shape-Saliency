from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import numpy as np

import sys, getopt, os
import math
from scipy.spatial import distance
import random
from sklearn import preprocessing
import h5py


from utils.util import *

# area in px as borders for each shape
area = 50
image_size = 400
kernel = 801
sigma = 32
verbose = 0 # Store images for view
store_verbose = '../images/dataset/'

# where to store h5py file
store_file = '../datasets/data.h5'
train_size = 0.8

## INFO: Script to extract all shapes and their respective saliency from each model and store it for NN learning
def extract_dataset():

    # Get info about models
    loc_models, glob_models = get_loc_glob_models_list()

    # Prepare h5py format for storage
    # Open file and prepare for append
    h5f = h5py.File(store_file, 'w')
    train_loc = h5f.create_dataset("train_loc", (0, image_size, image_size, len(loc_models) + 1),
                                   maxshape=(None, image_size, image_size, len(loc_models) + 1))
    valid_loc = h5f.create_dataset("valid_loc", (0, image_size, image_size, len(loc_models) + 1),
                                   maxshape=(None, image_size, image_size, len(loc_models) + 1))
    train_gt_loc = h5f.create_dataset("train_gt_loc", (0, image_size, image_size, 1),
                                   maxshape=(None, image_size, image_size, 1))
    valid_gt_loc = h5f.create_dataset("valid_gt_loc", (0, image_size, image_size, 1),
                                      maxshape=(None, image_size, image_size, 1))

    train_glob = h5f.create_dataset("train_glob", (0, image_size, image_size, len(glob_models) + 1),
                                   maxshape=(None, image_size, image_size, len(glob_models) + 1))
    valid_glob = h5f.create_dataset("valid_glob", (0, image_size, image_size, len(glob_models) + 1),
                                   maxshape=(None, image_size, image_size, len(glob_models) + 1))
    train_gt_glob = h5f.create_dataset("train_gt_glob", (0, image_size, image_size, 1),
                                      maxshape=(None, image_size, image_size, 1))
    valid_gt_glob = h5f.create_dataset("valid_gt_glob", (0, image_size, image_size, 1),
                                      maxshape=(None, image_size, image_size, 1))

    # Get list of images used in experiment
    images = get_images_list()

    # List of images to store later:
    store_image_loc = None
    store_image_glob = None
    store_gt_loc = None
    store_gt_glob = None
    store_image_loc = np.zeros(shape=(0, image_size, image_size, len(loc_models) + 1))
    store_image_glob = np.zeros(shape=(0, image_size, image_size, len(glob_models) + 1))
    store_gt_loc = np.zeros(shape=(0, image_size, image_size, 1))
    store_gt_glob = np.zeros(shape=(0, image_size, image_size, 1))

    # Extract basic shape image and its coordinates
    for i, image in enumerate(images):
        # For each contour:
        # 1. get coordinates
        # 2. store unedited shape
        # 3. store all local models
        # 4. store ground truth

        # Two images - continue due to certain models not having information about it
        if 'two' in image:
            continue

        # Load image and binarize it
        img = cv.imread(os.path.join('../images/.', image), 0)
        name = os.path.splitext(image)[0]

        # binarize it (just to be sure)
        ret, thr = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        # find contours of image
        # cv.CHAIN_APPROX_NONE -> we are looking for each boundary pixel
        # cv.RETR_EXTERNAL -> we are not looking for contours in contours
        im2, contours, hierarchy = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Ground truth for global saliency models # We can just compute it once
        blank_image = np.zeros((thr.shape[0], thr.shape[1]), np.float) #uint8
        gt_glob_image, scaled_gt_glob_image = get_global_gt(name, blank_image, max_fix=4, sigma=sigma, kernel=kernel,
                                                            binary=False)
        if verbose == 1:
            cv.imwrite(store_verbose + str(i) + '_gt_global_all.png', gt_glob_image)

        for j, cnt in enumerate(contours):
            if cnt.shape[0] < 30:
                # erase noise
                continue
            # For each contour store all information and resize it for NN
            max_x, min_x, max_y, min_y = get_contour_margins(cnt)

            bx, by, bw, bh = cv.boundingRect(cnt)

            #### LOCAL CONTEXT ####

            # area px on each side
            blank_image = np.zeros((max_y - min_y + 2 * area, max_x - min_x + 2 * area), np.float) #uint8
            shape_image, scaled_shape_image = get_clipped_shape(thr, bx, by, bw, bh, area, image_size, True)
            gt_loc_shape_image, scaled_gt_loc_shape_image = get_local_gt(name, blank_image, bx, by, bw, bh, max_fix=4, sigma=sigma,
                                                             kernel=kernel, binary=False, area=area, resize=image_size)
            # scaled image 0 - 1 for network.
            result_image = scaled_shape_image
            if verbose == 1:
                cv.imwrite(store_verbose + str(i) + '_' + str(j) + '_gt_local.png', gt_loc_shape_image)
                cv.imwrite(store_verbose + str(i) + '_' + str(j) + '_base_shape.png', shape_image)

            # extract each local model now:
            for model in loc_models:
                mod_img = cv.imread(os.path.join('../models/local/.', model, image), 0)
                mod_shape, scaled_mod_shape = get_clipped_shape(mod_img, bx, by, bw, bh, area, image_size, True)
                # Create multichannel image
                result_image = np.concatenate((result_image, scaled_mod_shape), axis=2)
                if verbose == 1:
                    cv.imwrite(store_verbose+str(i)+'_'+str(j)+'_loc_'+model+'.png',mod_shape)

            #store_image_loc.append(result_image)
            result_image = result_image.reshape((1,) + result_image.shape)
            store_image_loc = np.append(store_image_loc, result_image, axis=0)

            scaled_gt_loc_shape_image = scaled_gt_loc_shape_image.reshape(scaled_gt_loc_shape_image.shape + (1,))
            #store_gt_loc.append(scaled_gt_loc_shape_image)
            scaled_gt_loc_shape_image = scaled_gt_loc_shape_image.reshape((1,) + scaled_gt_loc_shape_image.shape)
            store_gt_loc = np.append(store_gt_loc, scaled_gt_loc_shape_image, axis=0)

            #### GLOBAL CONTEXT ####

            gt_glob_shape_image, scaled_gt_glob_shape_image = get_clipped_shape(gt_glob_image, bx, by, bw, bh, area, image_size, True)
            result_image = scaled_shape_image
            if verbose == 1:
                cv.imwrite(store_verbose + str(i) + '_' + str(j) + '_gt_global.png', gt_glob_shape_image)

            # extract each global model now:
            for model in glob_models:
                mod_img = cv.imread(os.path.join('../models/global/.', model, image), 0)
                mod_shape, scaled_mod_shape = get_clipped_shape(mod_img, bx, by, bw, bh, area, image_size, True)
                # Create multichannel image
                result_image = np.concatenate((result_image, scaled_mod_shape), axis=2)
                if verbose == 1:
                    cv.imwrite(store_verbose+str(i)+'_'+str(j)+'_glob_'+model+'.png',mod_shape)

            #store_image_glob.append(result_image)
            result_image = result_image.reshape((1,) + result_image.shape)
            store_image_glob = np.append(store_image_glob, result_image, axis=0)

            #store_gt_glob.append(scaled_gt_glob_shape_image)
            scaled_gt_glob_shape_image = scaled_gt_glob_shape_image.reshape((1,) + scaled_gt_glob_shape_image.shape)
            store_gt_glob = np.append(store_gt_glob, scaled_gt_glob_shape_image, axis=0)

    # APPEND INTO H5PY FILE
    # LOCAL
    # Split into training / valid datasets
    end_t = round(len(store_image_loc) * train_size)
    train_set = store_image_loc[1:end_t, :, :, :]
    valid_set = store_image_loc[end_t:, :, :, :]
    train_gt_set = store_gt_loc[1:end_t, :, :, :]
    valid_gt_set = store_gt_loc[end_t:, :, :, :]

    # Store as an append file
    # First reshape
    train_loc.resize(train_loc.shape[0] + len(train_set), axis=0)
    valid_loc.resize(valid_loc.shape[0] + len(valid_set), axis=0)

    train_gt_loc.resize(train_gt_loc.shape[0] + len(train_gt_set), axis=0)
    valid_gt_loc.resize(valid_gt_loc.shape[0] + len(valid_gt_set), axis=0)

    # Now append
    train_loc[-len(train_set):] = train_set
    valid_loc[-len(valid_set):] = valid_set
    train_gt_loc[-len(train_set):] = train_gt_set
    valid_gt_loc[-len(valid_set):] = valid_gt_set

    # GLOBAL
    # Split into training / valid datasets
    end_t = round(len(store_image_glob) * train_size)
    train_set = store_image_glob[1:end_t, :, :, :]
    valid_set = store_image_glob[end_t:, :, :, :]
    train_gt_set = store_gt_glob[1:end_t, :, :, :]
    valid_gt_set = store_gt_glob[end_t:, :, :, :]

    # Store as an append file
    # First reshape
    train_glob.resize(train_glob.shape[0] + len(train_set), axis=0)
    valid_glob.resize(valid_glob.shape[0] + len(valid_set), axis=0)

    train_gt_glob.resize(train_gt_glob.shape[0] + len(train_gt_set), axis=0)
    valid_gt_glob.resize(valid_gt_glob.shape[0] + len(valid_gt_set), axis=0)

    # Now append
    train_glob[-len(train_set):] = train_set
    valid_glob[-len(valid_set):] = valid_set
    train_gt_glob[-len(train_set):] = train_gt_set
    valid_gt_glob[-len(valid_set):] = valid_gt_set


if __name__ == "__main__":
    extract_dataset()