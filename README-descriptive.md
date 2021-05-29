# CMT316_Group_Project
CMT316 Group Project: Object localisation CV4

Authors: Edward Parkinson, Iwan Munro, Lewis Hemming, Shaoshi Sun

Raw dataset available at: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

The descriptive analysis is based on the partially pre-processed data. The script requires the images to be in three separate folders for train / test / val, and the label information to be in .txt files in three separate folders train / test / val. These are provided in the data and are the outputs of the initial pre-processing of the raw VOC2012 data.

The script requires two output folders:

../labels_csv
../report

Therefore, the expected folder within the root directory is:

../images/train
../images/val
../images/test
../labels/train
../labels/val
../labels/test
../labels_csv
../report

The script does the following:

- Process all of the label information in the txt files into three separate pandas dataframes, one each of train, val and test and store these as .csv files for future use.
These are stores in the ../labels_csv folder

- Use both the label information in the dataframes and the raw images to produce the four illustrations in the report. The output is 4 .jpg files which are stored in the ../report folder.

The script can be run in one go from the command prompt.
