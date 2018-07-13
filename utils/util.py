import cv2 as cv
import numpy as np
import sys, getopt, os
import math
from scipy.ndimage.filters import convolve, correlate
import scipy.signal

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
                #blank_image[int(math.floor(pt_y - by)) - 2 + area:int(math.floor(pt_y - by)) + 2 + area, int(math.floor(pt_x - bx)) - 2 + area:int(math.floor(pt_x - bx)) + 2 + area, 0] = 255
                blank_image[int(math.floor(pt_y - by)) + area, int(math.floor(pt_x - bx)) + area] = 255.0

    if resize > 0:
        blank_image = cv.resize(blank_image, (resize, resize))
        blank_image = blank_image.reshape(resize, resize, 1)

    if sigma > 0:
        k = matlab_style_gauss2D(shape=(kernel, kernel), sigma=sigma)
        #blank_image = blank_image.reshape(blank_image.shape[0], blank_image.shape[1])
        blank_image = cv.filter2D(blank_image, -1, k)
        #blank_image = blank_image.reshape(blank_image.shape[0], blank_image.shape[1], 1)

        # FIX: old OpenCV way:
        # blank_image = cv.GaussianBlur(blank_image, (int(kernel), int(kernel)), float(sigma))

        #scale image up to 255
        blank_image = blank_image * int(255. / blank_image.max())
        # scale 0 - 1 = 0 - 255 for network
        scaled_image = np.zeros((blank_image.shape[0], blank_image.shape[1]), np.float)
        delim = 1./255
        scaled_image = blank_image * delim

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
            #blank_image[int(math.floor(pt_y)) - 2:int(math.floor(pt_y)) + 2, int(math.floor(pt_x)) - 2:int(math.floor(pt_x)) + 2, 0] = 255
            blank_image[int(math.floor(pt_y)), int(math.floor(pt_x))] = 255.

    if sigma > 0:
        k = matlab_style_gauss2D(shape=(kernel, kernel), sigma=sigma)
        #blank_image = blank_image.reshape(blank_image.shape[0], blank_image.shape[1])
        blank_image = cv.filter2D(blank_image, -1, k)
        # blank_image = blank_image.reshape(blank_image.shape[0], blank_image.shape[1], 1)

        # FIX: old OpenCV way:
        # blank_image = cv.GaussianBlur(blank_image, (int(kernel), int(kernel)), float(sigma))

        #scale image up to 255
        blank_image = blank_image * int(255.0 / blank_image.max())

        # scale 0 - 1 = 0 - 255 for network
        scaled_image = np.zeros((blank_image.shape[0], blank_image.shape[1]), np.float)
        delim = 1./255
        scaled_image = blank_image * delim

    return blank_image, scaled_image

def get_clipped_shape(from_image, bx, by, bw, bh, area, resize, full):
    shape_image = np.zeros((bh + 2 * area, bw + 2 * area, 1), np.float)
    if full: # include also added area in clipping
        # check for borders
        b1 = (by - area) if (by - area) > 0 else 0
        e1 = (by + bh + area) if (by + bh + area) <= from_image.shape[0] else from_image.shape[0]
        b2 = (bx - area) if (bx - area) > 0 else 0
        e2 = (bx + bw + area) if (bx + bw + area) <= from_image.shape[1] else from_image.shape[1]

        if b1 == 0:
            sb1 = abs(by - area)
        else:
            sb1 = 0
        if e1 == from_image.shape[0]:
            se1 = e1 - b1
        else:
            se1 = 2 * area + bh
        if b2 == 0:
            sb2 = abs(bx - area)
        else:
            sb2 = 0
        if e2 == from_image.shape[1]:
            se2 = e2 - b2
        else:
            se2 = 2 * area + bw
        shape_image[sb1:se1, sb2:se2, 0] = from_image[b1:e1, b2:e2]
        #shape_image[0:2 * area + bh, 0:2 * area + bw, 0] = from_image[b1:e1, b2:e2]
    else:
        shape_image[area:area + bh, area:area + bw, 0] = from_image[by:by + bh, bx:bx + bw]

    if resize > 0:
        shape_image = cv.resize(shape_image, (resize, resize))
        shape_image = shape_image.reshape(resize, resize, 1)

    # We need to scale it to 0-1 based on whole image (in case of global saliency models)
    scaled_shape_image = np.zeros((shape_image.shape[0], shape_image.shape[1], 1), np.float)
    scaled_shape_image = shape_image * 1./255.

    return shape_image, scaled_shape_image

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def conv2(x,y,mode='same'):
    """
    Emulate the function conv2 from Mathworks.

    Usage:

    z = conv2(x,y,mode='same')

    TODO:
     - Support other modes than 'same' (see conv2.m)
    """

    if not(mode == 'same'):
        raise Exception("Mode not supported")

    # Add singleton dimensions
    if (len(x.shape) < len(y.shape)):
        dim = x.shape
        for i in range(len(x.shape),len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)
    elif (len(y.shape) < len(x.shape)):
        dim = y.shape
        for i in range(len(y.shape),len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce
    # the results of scipy.signal.convolve and Matlab
    for i in range(len(x.shape)):
        if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
             x.shape[i] > 1 and
             y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = convolve(x,y, mode='constant', origin=origin)

    return z