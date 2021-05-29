import os
import xml.etree.ElementTree as ET
from os import getcwd
from tqdm import tqdm
sets = [('2012', 'train'), ('2012', 'val'), ('2012', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# this function takes the raw bouding box data and normalises to between 0 and 1 and converts the description
# of the bounding box into x and y coordinates and box width and height
def convert_box(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

# this function parses each xml file and extracts the class name, then converst to to class ID based on the list above
# then pull out the x and y min / max of the bounding box
def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt' % (year, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert_box((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


cwd = getcwd()
for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/' % year):
        os.makedirs('VOCdevkit/VOC%s/labels/' % year)
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in tqdm(image_ids):
        list_file.write('VOCdevkit/VOC%s/JPEGImages/%s.jpg\n' % ( year, image_id))
        convert_annotation(year, image_id)
    list_file.close()



import os
if not os.path.exists('../VOC'):
    os.makedirs('../VOC')
    os.makedirs('../VOC/images')
    os.makedirs('../VOC/images/train')
    os.makedirs('../VOC/images/val')
    os.makedirs('../VOC/images/test')
    os.makedirs('../VOC/labels')
    os.makedirs('../VOC/labels/train')
    os.makedirs('../VOC/labels/val')
    os.makedirs('../VOC/labels/test')

#Train set
print(os.path.exists('VOCdevkit/VOC2012/ImageSets/Main/train.txt'))
with open('VOCdevkit/VOC2012/ImageSets/Main/train.txt', 'r') as f:
    for line in f.readlines():
        line = "/".join(line.split('/')[-5:]).strip()
        line+='.jpg'
        if os.path.exists("VOCdevkit\\VOC2012\\JPEGImages\\" + line):
            os.system("copy VOCdevkit\\VOC2012\\JPEGImages\\" + line + " ..\\VOC\\images\\train")
        line = line.replace('JPEGImages', 'labels').replace('jpg', 'txt')
        if os.path.exists("VOCdevkit\\VOC2012\\labels\\" + line):
            os.system("copy VOCdevkit\\VOC2012\\labels\\" + line + " ..\\VOC\\labels\\train")

#Validation set
print(os.path.exists('VOCdevkit/VOC2012/ImageSets/Main/val.txt'))
with open('VOCdevkit/VOC2012/ImageSets/Main/val.txt', 'r') as f:
    for line in f.readlines():
        line = "/".join(line.split('/')[-5:]).strip()
        line+='.jpg'
        if os.path.exists("VOCdevkit\\VOC2012\\JPEGImages\\" + line):
            os.system("copy VOCdevkit\\VOC2012\\JPEGImages\\" + line + " ..\\VOC\\images\\val")
        line = line.replace('JPEGImages', 'labels').replace('jpg', 'txt')
        if os.path.exists("VOCdevkit\\VOC2012\\labels\\" + line):
            os.system("copy VOCdevkit\\VOC2012\\labels\\" + line + " ..\\VOC\\labels\\val")

#Test set
print(os.path.exists('VOCdevkit/VOC2012/ImageSets/Main/test.txt'))
with open('VOCdevkit/VOC2012/ImageSets/Main/test.txt', 'r') as f:
    for line in f.readlines():
        line = "/".join(line.split('/')[-5:]).strip()
        line+='.jpg'
        if os.path.exists("VOCdevkit\\VOC2012\\JPEGImages\\" + line):
            os.system("copy VOCdevkit\\VOC2012\\JPEGImages\\" + line + " ..\\VOC\\images\\test")
        line = line.replace('JPEGImages', 'labels').replace('jpg', 'txt')
        if os.path.exists("VOCdevkit\\VOC2012\\labels\\" + line):
            os.system("copy VOCdevkit\\VOC2012\\labels\\" + line + " ..\\VOC\\labels\\test")
