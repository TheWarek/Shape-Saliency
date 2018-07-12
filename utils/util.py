import cv2 as cv
import numpy as np
import sys, getopt, os
import math


def get_loc_glob_models_list():
    # Get info about models

    loc_models = []
    glob_models = []

    with open('../models/info.txt', 'r') as file:
        line = True
        while line:
            line = file.readline()
            if not line:
                break
            line_split = line.split()

            # In case of new participant reset
            if line_split[0] == 'local:':
                loc_models.append(line_split[1])
            if line_split[0] == 'global:':
                glob_models.append(line_split[1])

    return loc_models, glob_models

def get_images_list():

    images = []
    for file in os.listdir('../images/.'):
        if file.endswith(".png"):
            name = os.path.splitext(file)[0]
            images.append(file)

    return images

def get_contour_margins(cnt):
    # find max / min point of the contour
    max_x = np.max([a[0][0] for a in cnt])
    min_x = np.min([a[0][0] for a in cnt])
    max_y = np.max([a[0][1] for a in cnt])
    min_y = np.min([a[0][1] for a in cnt])

    return max_x, min_x, max_y, min_y

def get_local_gt(name, blank_image, bx, by, bw, bh, max_fix, sigma, kernel, binary, area, resize):

    # scaler = MinMaxScaler(copy=True, feature_range=(0, 255))
    # scaler.fit()

    with open('../fixations/' + name + '.txt', 'r') as file:
        line = True
        max_i = 0
        while line:
            line = file.readline()
            if not line:
                break
            line_split = line.split()
            # In case of new participant reset
            if len(line_split) <= 1:
                max_i = 0
                continue
            # in case of overcoming the number of fixation for current participant continue
            if max_i >= max_fix:
                continue
            pt_x = float(line_split[3])
            pt_y = float(line_split[4])

            # In case of fixation within bounding area:
            if (bx <= pt_x <= bx + bw) and (by <= pt_y <= by + bh):
                max_i = max_i + 1
                blank_image[int(math.floor(pt_y - by)) - 2 + area:int(math.floor(pt_y - by)) + 2 + area, int(math.floor(pt_x - bx)) - 2 + area:int(math.floor(pt_x - bx)) + 2 + area, 0] = 255
                # cv.imshow('saliency', blank_image)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

    if resize > 0:
        blank_image = cv.resize(blank_image, (resize, resize))
        blank_image = blank_image.reshape(resize, resize, 1)

    if sigma > 0:
        blank_image = cv.GaussianBlur(blank_image, (int(kernel), int(kernel)), float(sigma))
        #scale image up to 255
        blank_image = blank_image * int(255 / blank_image.max())
        # scale 0 - 1 = 0 - 255 for network
        scaled_image = np.zeros((blank_image.shape[0], blank_image.shape[1], 1), np.float64)
        delim = 1./255
        scaled_image = blank_image * delim
        #blank_image = scaler.transform(blank_image)
    return blank_image, scaled_image

def get_global_gt(name, blank_image, max_fix, sigma, kernel, binary):
    with open('../fixations/' + name + '.txt', 'r') as file:
        line = True
        max_i = 0
        while line:
            line = file.readline()
            if not line:
                break
            line_split = line.split()
            # In case of new participant reset
            if len(line_split) <= 1:
                max_i = 0
                continue
            # in case of overcoming the number of fixation for current participant continue
            if max_i >= max_fix:
                continue
            pt_x = float(line_split[3])
            pt_y = float(line_split[4])

            # In case of fixation:
            max_i = max_i + 1
            blank_image[int(math.floor(pt_y)) - 2:int(math.floor(pt_y)) + 2, int(math.floor(pt_x)) - 2:int(math.floor(pt_x)) + 2, 0] = 255

    if sigma > 0:
        blank_image = cv.GaussianBlur(blank_image, (int(kernel), int(kernel)), float(sigma))
        #scale image up to 255
        blank_image = blank_image * int(255 / blank_image.max())
        # scale 0 - 1 = 0 - 255 for network
        scaled_image = np.zeros((blank_image.shape[0], blank_image.shape[1], 1), np.float64)
        delim = 1./255
        scaled_image = blank_image * delim
        #blank_image = scaler.transform(blank_image)
    return blank_image, scaled_image

def get_clipped_shape(from_image, bx, by, bw, bh, area, resize):
    shape_image = np.zeros((bh + 2 * area, bw + 2 * area, 1), np.uint8)
    shape_image[area:area + bh, area:area + bw, 0] = from_image[by:by + bh, bx:bx + bw]

    if resize > 0:
        shape_image = cv.resize(shape_image, (resize, resize))
        shape_image = shape_image.reshape(resize, resize, 1)

    # We need to scale it to 0-1 based on whole image (in case of global saliency models)
    scaled_shape_image = np.zeros((shape_image.shape[0], shape_image.shape[1], 1), np.float64)
    scaled_shape_image = shape_image * 1./255.

    return shape_image, scaled_shape_image