import cv2
import os
import re
import csv
import numpy as np


main_path = '../images/'
paths = ['train','test','val']

for p in paths:
    for file in os.listdir((main_path + p)):
        img = cv2.imread((main_path + p + '/' + file))
        np.save('numpy_images/' + p + '/' + file[0:-4] + '.npy', img)
