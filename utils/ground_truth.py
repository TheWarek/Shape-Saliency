from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import numpy as np

import sys, getopt, os
import math
from scipy.spatial import distance
import random
from sklearn.preprocessing import MinMaxScaler


## Pipeline:
# - find contours
# - Create ROI
# - Open fixation map for each participant
# - For each area take X number of fixations and make ground truth map

def parse_area(name, blank_image, bx, by, bw, bh, max_fix, sigma, kernel, binary, area):

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

def extract_ground_truth(fixations, sigma, kernel, binary, resize, width, heigth, area):
    # load images

    im_counter = 0

    for file in os.listdir('../images/.'):
        if file.endswith(".png"):
            img = cv.imread(os.path.join('../images/.', file), 0)
            img_color = cv.imread(os.path.join('../images/.', file))

            name = os.path.splitext(file)[0]
            # binarize it (just to be sure)
            ret, thr = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
            # find contours of image
            # cv.CHAIN_APPROX_NONE -> we are looking for each boundary pixel
            # cv.RETR_EXTERNAL -> we are not looking for contours in contours
            im2, contours, hierarchy = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            for cnt in contours:
                if cnt.shape[0] < 30:
                    # erase noise
                    continue

                # # center of mass
                # M = cv.moments(cnt)
                # cx = int(M['m10'] / M['m00'])
                # cy = int(M['m01'] / M['m00'])

                # find max / min point of the contour
                max_x = np.max([a[0][0] for a in cnt])
                min_x = np.min([a[0][0] for a in cnt])
                max_y = np.max([a[0][1] for a in cnt])
                min_y = np.min([a[0][1] for a in cnt])

                # create empty image for each shape
                # TODO predefined size for later deep learning ??

                bx, by, bw, bh = cv.boundingRect(cnt)

                # area px on each side
                blank_image = np.zeros((max_y - min_y + 2 * area, max_x - min_x + 2 * area, 1), np.uint8)
                shape_image = np.zeros((max_y - min_y + 2 * area, max_x - min_x + 2 * area, 1), np.uint8)
                shape_image[area:area+bh, area:area+bw, 0] = thr[by:by+bh, bx:bx+bw]



                blank_image, scaled_image = parse_area(name, blank_image, bx, by, bw, bh, fixations, sigma, kernel, binary, area)

                cv.rectangle(img_color, (bx,by), (bx+bw,by+bh), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                cv.drawContours(img_color, [cnt], 0, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 3)

                # cv.imshow('saliency', blank_image)
                # cv.imshow('shape', shape_image)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

                # store images
                cv.imwrite('../images/local_gt/'+str(im_counter)+'_gt.png', blank_image)
                cv.imwrite('../images/local_gt/'+str(im_counter)+'_shape.png', shape_image)
                cv.imwrite('../images/local_gt/'+str(im_counter)+'_gt_scaled.png', scaled_image)
                im_counter = im_counter + 1



            # cv.imshow('test',img_color)
            # cv.waitKey(0)
            # cv.destroyAllWindows()



def main(argv):
    fixations = 4
    sigma = 32
    binary = False
    resize = False
    width = 400
    heigth = 400
    kernel = 3
    area = 50
    try:
      opts, args = getopt.getopt(argv, "hf:s:k:ba:", ["fixations=", "sigma=", "kernel=", "area="])
    except getopt.GetoptError:
      print('ground_truth.py -f <number_of_fixations> -s <sigma> -k <kernel_size> -b (if set then binary) -r (if to resize) -w <width> -h <heigth> -a <margin_area>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('ground_truth.py -f <number_of_fixations> -s <sigma> -k <kernel_size> -b (if set then binary) -r (if to resize) -w <width> -h <heigth> -a <margin_area>')
         sys.exit()
      elif opt == '-f':
          fixations = int(arg)
      elif opt == '-t':
          time = arg
      elif opt == '-k':
          kernel = arg
      elif opt == '-b':
          binary = True
      elif opt == '-r':
          resize = True
      elif opt == '-w':
          width = arg
      elif opt == '-h':
          heigth = arg
      elif opt == '-a':
          area = arg
    extract_ground_truth(fixations, sigma, kernel, binary, resize, width, heigth, area)

if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])