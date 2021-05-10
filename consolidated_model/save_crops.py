# To save object crops based upon pixel range in different folders
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
viz_dir = ann_dir.replace('annotations', 'crop_imgs')
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)

with open("class_list.txt") as f:
    class_list = [line.rstrip() for line in f]
#print(class_list)

with open("color_list.txt") as f:
    color_list = [line.rstrip() for line in f]
#print(color_list)

#json_file = open('json_list.txt', 'w')

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
            category_dict[img_path].append([_id, name])
        #print(category_dict)
        obj_list = category_dict[img_path][1:]
        #print(obj_list)
        for annotation in annotations:
            _id = annotation['id']
            y_min = annotation['bbox']['y_min']
            y_max = annotation['bbox']['y_max']
            x_min = annotation['bbox']['x_min']
            x_max = annotation['bbox']['x_max']
            bboxw = x_max - x_min
            bboxh = y_max - y_min
            if bboxh == 0 or bboxw == 0:
                continue
            class_name = ''
            for obj in obj_list:
                if _id == obj[0]:
                    class_name = obj[1]
            #print(_id, class_name, y_min, y_max, x_max, x_min)
            if class_name in class_list:
                idx = class_list.index(class_name)
                color = color_list[idx]
                color = eval(color)
            else: 
                color = (128, 255, 0)
                #color = list(np.random.choice(range(255), size=3))
                #color = tuple(color)
            if cv_img is None:
                continue
            crop_img = cv_img[y_min:y_max, x_min:x_max]
            crop_name = img_name.split('.jpg')[0] + '_' + class_name + '_' + str(_id) + '.jpg'
            #print(crop_name, bboxw, bboxh)
            class_dir = viz_dir + '/' + class_name
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            if bboxw < 10 or bboxh < 10:
                pix_dir = class_dir + '/' + '10px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)
            elif bboxw < 20 or bboxh < 20:
                pix_dir = class_dir + '/' + '20px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)
            elif bboxw < 30 or bboxh < 30:
                pix_dir = class_dir + '/' + '30px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)                    
            elif bboxw < 40 or bboxh < 40:
                pix_dir = class_dir + '/' + '40px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)                    
            elif bboxw < 50 or bboxh < 50:
                pix_dir = class_dir + '/' + '50px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)                    
            elif bboxw < 60 or bboxh < 60:
                pix_dir = class_dir + '/' + '60px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)
            elif bboxw < 70 or bboxh < 70:
                pix_dir = class_dir + '/' + '70px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)                    
            elif bboxw < 80 or bboxh < 80:
                pix_dir = class_dir + '/' + '80px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)
            elif bboxw < 90 or bboxh < 90:
                pix_dir = class_dir + '/' + '90px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)
            elif bboxw < 100 or bboxh < 100:
                pix_dir = class_dir + '/' + '100px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)
            elif bboxw >= 100 or bboxh >= 100:
                pix_dir = class_dir + '/' + 'gt_100px'
                if not os.path.exists(pix_dir):
                    os.makedirs(pix_dir)                    
            #json_file.write()
            if crop_img is not None:
                cv2.imwrite(pix_dir + '/' + crop_name, crop_img)
    #sys.exit()        
            #cv2.rectangle(cv_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            #cv2.putText(cv_img, class_name, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX,  
            #       1, color, 2, cv2.LINE_AA)
        #if cv_img is not None:
        #    cv2.imwrite(viz_dir + '/' + img_name, cv_img) 
        #sys.exit()
