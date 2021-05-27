import cv2
import os
import re
import csv
import numpy as np


main_path = '../images\\'
paths = ['train','test','val']

for p in paths:
    for file in os.listdir((main_path + p)):
        for existing_file in os.listdir(('CSV_Images/' + p)):
            if file[0:-4] != existing_file[0:-4]:
            print(file)
            csv_name = 'CSV_Images/' + p + '/' + file[0:-3] + 'csv'
            img = cv2.imread((main_path + p + '/' + file))
            lst = img.tolist()
            with open(csv_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerows(lst)

'''
Reading
--------------------------------------------
with open(file_name+'.csv', 'r') as f:
  reader = csv.reader(f)
  examples = list(reader)

print(examples)
nwexamples = []
for row in examples:
    nwrow = []
    for r in row:
        nwrow.append(eval(r))
    nwexamples.append(nwrow)
print(nwexamples)
'''