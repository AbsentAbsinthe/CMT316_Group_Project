import torchvision
import torch
import os
import sys
import csv
import cv2
from torchvision import transforms

train_images_path = "images/train/"
test_images_path = "images/test/"
train_labels_path = "labels_csv/df_labels_train.csv"
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

# Image Preprocessing
model.train()
train_images = []
for image in os.listdir(train_images_path):
    img = cv2.imread(train_images_path + image)
    trans = transforms.Compose([transforms.ToTensor()])
    norm_img = trans(img)
    train_images.append(norm_img)

print('Image Preprocessing Complete...')

# Label Preprocessing 
reader = csv.reader(open(train_labels_path, 'r'))
labels_in = list(reader)
label_lists = []
for label in labels_in[1:]:
    all_boxes = []
    all_labels = []
    for coords in range(2,len(label),5):
        boxes = []
        labels = []
        if len(label[coords:]) > 4:
            if len(label[coords]) > 0 and len(label[coords+1]) > 0 and len(label[coords+2]) > 0 and len(label[coords+3]) > 0 and len(label[coords+ 4]) > 0:
                labels.append(float(label[coords]))
                boxes.append(float(label[coords + 1]))
                boxes.append(float(label[coords + 2]))
                boxes.append((float(label[coords + 1]) + float(label[coords + 3])))
                boxes.append((float(label[coords + 2]) + float(label[coords + 4])))
        if len(boxes) != 0:
            all_boxes.append(boxes)
            all_labels.append(labels)
    box_tensors = torch.tensor(all_boxes)
    label_tensors = torch.tensor(all_labels)
    label_lists.append([box_tensors,label_tensors])

print('Label Preprocessing Complete...')

# Model Training
targets = []
for i in range(len(train_images)):
    d = {}
    d['boxes'] = label_lists[i][0]
    d['labels'] = label_lists[i][1]
    targets.append(d)
output = model(train_images, targets)

print('Training Complete...')

# Model Evaluation and Predictions
model.eval()
test_images = []
for image in os.listdir(test_images_path):
    img = torchvision.io.read_image(test_images_path + image)
    test_images.append(img)
predictions = model(test_images)
print('Predictions Complete...')
