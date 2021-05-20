import torchvision
import torch
import os
train_images_path = "images/train/"
test_images_path = "images/test/"
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
# For training
model.train()
train_images = []
for image in os.listdir(train_images_path):
    img = torchvision.io.read_image(train_images_path + image)
    train_images.append(img)

targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
output = model(images, targets)
# For inference
model.eval()
test_images = []
for image in os.listdir(test_images_path):
    img = torchvision.io.read_image(test_images_path + image)
    test_images.append(img)
predictions = model(test_images)