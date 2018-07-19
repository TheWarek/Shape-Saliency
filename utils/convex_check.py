from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import numpy as np

import sys, getopt, os
from scipy.spatial import distance
import random

def run_convex(level, sigma, kernel):
    # load images
    for file in os.listdir('../images/shapes/.'):
        if file.endswith(".png"):
            img = cv.imread(os.path.join('../images/shapes/.', file), 0)
            img_color = cv.imread(os.path.join('../images/shapes/.', file))
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
                # center of mass
                M = cv.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # find max / min point of the contour
                max_x = np.max([a[0][0] for a in cnt])
                min_x = np.min([a[0][0] for a in cnt])
                max_y = np.max([a[0][1] for a in cnt])
                min_y = np.min([a[0][1] for a in cnt])

                # create empty image for each shape
                # TODO predefined size for later deep learning ??
                blank_image = np.zeros((100, 100, 1), np.uint8)

                # TODO Theory behind local convex / concave saliency ??
                # Further from center of mass - bigger saliency ?
                # Ratio IN / OUT = Bigger difference -> bigger saliency ?
                # (IN = how many points in local neighbourhood of the contour point belong to the shape)
                # All done in spatial pyramid and distance dependent from the original local point

                # compute distance from center of mass for each point of contour
                distances = []
                for point in cnt:
                    distances.append(distance.euclidean(point[0], [cx, cy]))

                ### cv.drawContours(img_color, [cnt], 0, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 3)

            cv.imshow('test',img_color)
            cv.waitKey(0)
            cv.destroyAllWindows()



def main(argv):
    level = 2
    sigma = 0.5
    kernel = 3
    try:
      opts, args = getopt.getopt(argv, "hl:s:k:", ["levels=", "sigma=", "kernel="])
    except getopt.GetoptError:
      print('run.py -l <levels_in_pyramid> -s <sigma_for_gauss> -k <kernel_size>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('run.py -l <levels_in_pyramid> -s <sigma_for_gauss> -k <kernel_size>')
         sys.exit()
      elif opt == '-l':
          level = arg
      elif opt == '-s':
          sigma = arg
      elif opt == '-k':
          kernel = arg
    run_convex(level, sigma, kernel)

if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])