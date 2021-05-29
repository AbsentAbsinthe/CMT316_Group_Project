# install package if necessary
# pip install dataframe-image
# pip install bs4
# pip install html5lib
# pip install lxml

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
img_dir = str(file_path)+'images/'          # images
rpt_dir = str(file_path)+'report/'          # report output


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


# function to count the number of images based on aspect ratio

def find_image_sizes(image_dirs):
  """ file path -> ordered dict, list
  Return and ordered dictionary with image aspect ratios as keys and count as values
  for the images stored in the specified directory. In addition, return list of tuples
  of the dictionary key-value pairs, sorted in descenting order by values, i.e. the count
  of the number of images with that aspect ratio.
  """
  dict_image_size = OrderedDict()
  for dir in image_dirs:
    for file in os.listdir(dir):
      img = image.load_img(str(dir)+str(file))
      size = img.size
      if size not in dict_image_size: dict_image_size[size]=1
      else: dict_image_size[size]+=1
  sorted_image_size = sorted(dict_image_size.items(), key=lambda x: x[1], reverse=False)
  sorted_image_size_desc = sorted(sorted_image_size, key=operator.itemgetter(1), reverse=False)
  return dict_image_size, sorted_image_size_desc


# create dataframes  for train / val / test sets and save to csv to avoid the need to re-process later

df_labels_train = label_dataframe(str(lab_dir)+'train/')
df_labels_train.to_csv(str(lab_dir_csv)+'df_labels_train.csv')

df_labels_val = label_dataframe(str(lab_dir)+'val/')
df_labels_val.to_csv(str(lab_dir_csv)+'df_labels_val.csv')

df_labels_test = label_dataframe(str(lab_dir)+'test/')
df_labels_test.to_csv(str(lab_dir_csv)+'df_labels_test.csv')


# importing saved csv files with label dataframe and concatonate for train+val+test for analysis
df_train = pd.read_csv(str(lab_dir_csv)+'df_labels_train.csv')
df_val = pd.read_csv(str(lab_dir_csv)+'df_labels_val.csv')
df_test = pd.read_csv(str(lab_dir_csv)+'df_labels_test.csv')
df_full = pd.concat([df_train, df_val, df_test])

# summarise train, val and test sets in summary table and save formatted dataframe as jpeg
df_train_sum = df_train.groupby(['class_0']).count().rename(columns = {'Unnamed: 0': 'class_total'})
df_val_sum = df_val.groupby(['class_0']).count().rename(columns = {'Unnamed: 0': 'class_total'})
df_test_sum = df_test.groupby(['class_0']).count().rename(columns = {'Unnamed: 0': 'class_total'})
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "total"]
df_data_sum = df_train_sum.iloc[:,0:1].rename(columns = {'class_total':'train'}).rename_axis('class_id')
df_data_sum['val'] = df_val_sum['class_total']
df_data_sum['test'] = df_test_sum['class_total']
df_data_sum['total'] = df_data_sum['train']+df_data_sum['val']+df_data_sum['test']
df_data_sum.loc["total"] = df_data_sum.sum()
df_data_sum.insert(1, 'train_%', df_data_sum['train'].div(df_data_sum['train']['total']).round(decimals=3))
df_data_sum.insert(3, 'val_%', df_data_sum['val'].div(df_data_sum['val']['total']).round(decimals=3))
df_data_sum.insert(5, 'test_%', df_data_sum['test'].div(df_data_sum['test']['total']).round(decimals=3))
df_data_sum.insert(7, 'total_%', df_data_sum['total'].div(df_data_sum['total']['total']).round(decimals=3))
df_data_sum.insert(0, 'class_name', classes)
df_data_sum = df_data_sum.sort_values('total_%', ascending=False)
df_data_sum = pd.concat([df_data_sum.drop('total', axis=0), df_data_sum.iloc[[0],:]], axis=0)
format_dict = {'train':'{0:,.0f}', 'train_%': '{:.1%}',
               'val':'{0:,.0f}', 'val_%': '{:.1%}',
               'test':'{0:,.0f}', 'test_%': '{:.1%}',
               'total':'{0:,.0f}', 'total_%': '{:.1%}'
               }
df_data_sum_styled = df_data_sum.style.format(format_dict).hide_index()
dfi.export(df_data_sum_styled,
           str(rpt_dir)+'df_styled.jpg',
           table_conversion='matplotlib')

# create bar chart showing distribution of images by the class of the primary object

df_image_dist = df_data_sum.loc[:, ['total', 'class_name']].drop(['total'], axis=0).sort_values('total', ascending=True)
class_n = df_image_dist['class_name'].tolist()
totals = df_image_dist['total'].tolist()
y_pos = [i for i, _ in enumerate(class_n)]
plt.barh(y_pos, totals, color='green')
plt.ylabel("Primary Object Class")
plt.xlabel("Number of Images")
plt.title("Distribution of Images by Primary Object Class")
plt.yticks(y_pos, class_n)
plt.savefig(str(rpt_dir)+'bar_channels.jpg', dpi=300, bbox_inches='tight')

# create histogram of the distribution of number of objects by image

plt.hist(df_full["num_objects"].values, bins=40)
plt.title("Histogram: distribution of number of objects per image")
plt.ylabel("Frequency")
plt.xlabel("Objects per Image")
plt.savefig(str(rpt_dir)+'hist_objects.jpg', dpi=300, bbox_inches='tight')

print('The maximum number of objects in any image is:   '+str(max(df_full["num_objects"].values)))
print('The number of images with just one object is:    '+str(len(df_full[df_full.num_objects.eq(1)])))
print('The % of images with just one object is:   '+str(len(df_full[df_full.num_objects.eq(1)])/11540))

# create bar chart of number of images by image aspect ratio

image_dirs = [str(img_dir)+'train/', str(img_dir)+'val/', str(img_dir)+'test/']
dict_image_size, sorted_image_size_desc = find_image_sizes(image_dirs)
bars = 10
img_size_n = [i[0] for i in sorted_image_size_desc]
totals = [i[1] for i in sorted_image_size_desc]
n_bars = len(dict_image_size.keys()) - bars
y_pos = [i for i, _ in enumerate(img_size_n[n_bars:])]
plt.barh(y_pos, totals[n_bars:], color='green')
plt.title("{} Most Frequent Images Sizes".format(bars))
plt.ylabel("Image Size")
plt.xlabel("Number of Images")
plt.yticks(y_pos, img_size_n[n_bars:])

print('The total number of images processed is:  '+str(sum(totals)))
print('The total number of unique image sizes is:  '+str(len(dict_image_size.keys())))
plt.savefig(str(rpt_dir)+'aspect_ratio.jpg', dpi=300, bbox_inches='tight')
