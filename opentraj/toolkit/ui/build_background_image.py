# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import glob
import os
import cv2
import numpy as np


def make_bg_image_from_screenshots(im_files):
    im_sum = None

    for im_file in im_files:
        im_i = cv2.imread(im_file)
        if im_sum is None:
            im_sum = im_i.astype(np.float)
        else:
            im_sum += im_i.astype(np.float)

    im_sum = (im_sum/len(im_files)).astype(np.uint8)
    cv2.imshow("bg", im_sum)
    cv2.waitKeyEx()
    return im_sum


if __name__ == "__main__":
    eth_hotel_dir = "/home/cyrus/Pictures/ETH-Hotel"
    files = glob.glob(eth_hotel_dir + "/*.png")
    im_avg = make_bg_image_from_screenshots(files)
    cv2.imwrite(os.path.join(eth_hotel_dir, "avg.jpg"), im_avg)
