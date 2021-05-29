# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import operator
from collections import OrderedDict
import matplotlib
from PIL import Image
import csv
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import keras.preprocessing.image as image
import dataframe_image as dfi


# file paths of input data and output data and report illustrations

file_path = '/Users/Ed/Documents/Documents/AI/ml_applications/Coursework2/CV4_project/' # CHANGE THIS
lab_dir = str(file_path)+'labels/'          # labels
lab_dir_csv = str(file_path)+'labels_csv/'  # labels data in csv files


# function to process the .txt files with label information into a Pandas DataFrame

def label_dataframe(lab_dir):
  """ file path -> pandas dataframe
  Return a pandas dataframe containing the image annotation data contained in the
  text files in the specified directory. for each line in a text file, add five items
  to the row in the dataframe for that image, object class, x, y, w h of the bounding box
  """
  df_labels = []
  for file in os.listdir(lab_dir):
    row_labels = OrderedDict()
    num_objects = 0
    row_labels["fileID"] = str(file).split('.')[0]
    with open(str(lab_dir) + str(file)) as file_in:
      for i, line in enumerate(file_in):
        line_list = line.split(sep=' ')
        object_dict = OrderedDict()
        object_dict["class_{}".format(i)] = line_list[0].strip()
        object_dict["x_coord_{}".format(i)] = line_list[1].strip()
        object_dict["y_coord_{}".format(i)] = line_list[2].strip()
        object_dict["width_{}".format(i)] = line_list[3].strip()
        object_dict["height_{}".format(i)] = line_list[4].strip()
        row_labels.update(object_dict)
        num_objects += 1
    row_labels['num_objects'] = num_objects
    df_labels.append(row_labels)
  df_labels = pd.DataFrame(df_labels).sort_values(by='fileID')
  return (df_labels)

# create dataframes  for train / val / test sets and save to csv

df_labels_train = label_dataframe(str(lab_dir)+'train/')
df_labels_train.to_csv(str(lab_dir_csv)+'df_labels_train.csv')

df_labels_val = label_dataframe(str(lab_dir)+'val/')
df_labels_val.to_csv(str(lab_dir_csv)+'df_labels_val.csv')

df_labels_test = label_dataframe(str(lab_dir)+'test/')
df_labels_test.to_csv(str(lab_dir_csv)+'df_labels_test.csv')
