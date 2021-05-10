# select images which classes are in the contaminant.txt file and form a images list file
import os, sys
import json
import glob
import csv
import cv2
import numpy as np
from collections import defaultdict

ann_dir = sys.argv[1]
all_classes = []
img_dir = ann_dir.replace('annotations', 'img')
viz_dir = ann_dir.replace('annotations', 'img_viz_new')
#if not os.path.exists(viz_dir):
#    os.makedirs(viz_dir)

with open("class_list.txt") as f:
    class_list = [line.rstrip() for line in f]
#print(class_list)

with open("contaminant.txt") as f:
    contaminant_list = [line.rstrip() for line in f]
print(contaminant_list)
img_file = open('side_load_contaminant_img_list.txt', 'w')

for filename in glob.glob(ann_dir + "*.json"):
    with open(filename) as f:
        data = json.load(f)
        annotations = data['annotations']
        categories = data['categories']
        img_name = data['images'][0]['file_name']
        img_path = img_dir + '/' + img_name
        category_dict = defaultdict(list)
        print(img_path)
        cv_img = cv2.imread(img_path)
        category_dict[img_path].append([img_name])
        for category in categories:
            _id = category['id']
            try:
                name = category['name'] + '_' + category['subcategory']
            except:
                name = category['name']
            if name in contaminant_list:
                img_path = viz_dir + '/' + img_name
                img_file.write(img_path + '\n')
                break
            #sys.exit()