import cv2
import os
import re
import csv
import numpy as np


main_path = 'images\\'
paths = ['train','test','val']

for p in paths:
    for file in os.listdir((main_path + p)):
        img = cv2.imread((main_path + p + '/' + file))
        numpy.save('numpy_Images/' + p + '/' + file[0:-3] + '.npy')
