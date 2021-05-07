#!/usr/bin/python

import glob
import os,sys
import shutil
import cv2
#import Image

imageDir = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/cleaned_trainset/img_19C/'
labelsDir = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/cleaned_trainset/img_19C/'
tagList = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/cleaned_trainset/train_list/img_cl28.txt'
imageList = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/cleaned_trainset/train_list/train_19C_cl28.txt'

#imageDir = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/subset_collection/subset_3_aug_blackout_class_iou/train/img/'
#labelsDir = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/cleaned_trainset/subset_collection/subset_3_aug_blackout_class_iou/train/labels/'
#tagList = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/subset_collection/subset_3_aug_blackout_class_iou/subset_list/cl28_subset_3_train_tags.txt'
#imageList = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/subset_collection/subset_3_aug_blackout_class_iou/subset_list/cl28_subset_3_train_img.txt'
# list of classes to ignore
ignore_list = [2, 12, 13]
class_map = {
'0': '0',
'1': '1',
'3': '2',
'4': '3',
'5': '4',
'6':'5',
'7':'6',
'8':'7',
'9':'8',
'10':'9',
'11':'10',
'14':'11',
'15':'12',
'16':'13',
'17':'14',
'18':'15',
'19':'16',
'20':'17',
'21':'18',
}

tagListFile = open(tagList, 'r')
imageListFile = open(imageList, 'w')
imagesprocessed = 0
for line in tagListFile:
    line = line.strip('\n')
    img_path = line.split(' ')[0]
    tags = line.split(' ')[1:]
    print(img_path)
    img_path = img_path.replace('img', 'img_19C')
    img = cv2.imread(img_path)
    imgH, imgW, _ = img.shape
    labelFile = img_path.split('.jpg')[0] + '.txt'
    #labelFile = open(labelFile.replace('img', 'labels') , "w")
    labelFile = open(labelFile, "w")
    label_exists = False
    for tag in tags:
        x1, y1, x2, y2, class_id = tag.split (",")
        if int(class_id) not in ignore_list:
            new_cl_id = class_map[class_id]
            #print('old: ', class_id, 'new: ', new_cl_id)
            x = int(x1)
            y = int(y1)
            w = int(x2) - int(x1)
            h = int(y2) - int(y1)
            Cx = ( ( x + ( ( w * 1.) / 2 ) ) * 1. ) / imgW
            Cy = ( ( y + ( ( h * 1.) / 2 ) ) * 1.) / imgH
            iW = ( w * 1.) / imgW
            iH = ( h * 1.) / imgH
            labelFile.write(str(new_cl_id)+ " "+ str(Cx) + " " + str(Cy) + " " + str(iW) + " " + str(iH) + '\n')
            label_exists = True
    labelFile.close()
    if label_exists:
        imageListFile.write(img_path+'\n')
