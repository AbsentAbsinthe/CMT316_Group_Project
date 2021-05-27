import torchvision
import torch
import os
import csv
train_images_path = "CMT316_Group_Project/images/train/"
test_images_path = "CMT316_Group_Project/images/test/"
train_labels_path = "CMT316_Group_Project/labels_csv/df_labels_train.csv"
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
# For training
model.train()
train_images = []
for image in os.listdir(train_images_path):
    img = torchvision.io.read_image(train_images_path + image)
    #img = img.unsqueeze(1) 4D Tensor
    train_images.append(img)

reader = csv.reader(open(train_labels_path, 'r'))
labels_in = list(reader)
label_lists = []
for label in labels_in[1:]:
    boxes = []
    labels = []
    for coords in range(2,len(label),5):
        if len(label[coords:]) > 4:
            if len(label[coords]) > 0 and len(label[coords+1]) > 0 and len(label[coords+2]) > 0 and len(label[coords+3]) > 0 and len(label[coords+ 4]) > 0:
                labels.append(float(label[coords]))
                boxes.append(float(label[coords + 1]))
                boxes.append(float(label[coords + 2]))
                boxes.append((float(label[coords + 1]) + float(label[coords + 3])))
                boxes.append((float(label[coords + 2]) + float(label[coords + 4])))
    label_lists.append([torch.tensor(boxes),torch.tensor(labels)])
print(label_lists[1])
targets = []
for i in range(len(train_images)):
    d = {}
    d['boxes'] = label_lists[i][0]
    d['labels'] = label_lists[i][1]
    targets.append(d)
output = model(train_images, targets)
# For inference
model.eval()
test_images = []
for image in os.listdir(test_images_path):
    img = torchvision.io.read_image(test_images_path + image)
    test_images.append(img)
predictions = model(test_images)
