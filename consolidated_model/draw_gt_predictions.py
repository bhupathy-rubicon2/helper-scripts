import os, sys
import shutil
import cv2
import PIL
import csv
from PIL import Image
from PIL import ImageDraw
from collections import defaultdict, OrderedDict

det_csv_path = sys.argv[1]
dest_folder = sys.argv[2]
#Img folder path
root_path = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/testset/img/'
gt_labels = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/trainings/cl-8/detection/ground-truth/'
tagList = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/testset/test_list/img.txt'
class_dict = {'0':'bag_paper',
'1': 'bag_plastic',
'2': 'bike',
'3': 'bin_neg',
'4': 'bin_pos',
'5': 'bottle_glass',
'6':'bottle_plastic',
'7':'bus',
'8':'can',
'9':'car',
'10':'clamshell_food_plastic',
'11':'clamshell_food_styrofoam',
'12':'mailbox',
'13':'motorbike',
'14':'paper_waste',
'15':'person',
'16':'plastic_film',
'17':'styrofoam_block',
'18':'truck',
'19':'waste_container',
'20':'wood_processed',
'21':'wood_yard_waste'}
tag_list = [0,1,5,6,8,10,11,14,16,17]
csv_file = open(det_csv_path)
csv_reader = csv.reader(csv_file, delimiter=',')
old_name = ''
det_dict = defaultdict(list)
for row in csv_reader:
    if (len(row) == 8):
        img_name, class_id, x1, y1, x2, y2, confidence = row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        label_file = gt_labels + img_name.split('.jpg')[0] + '.txt'
        img_path = root_path + img_name
        det_dict[img_name].append([class_id, x1, y1, x2, y2, confidence])

tagListFile = open(tagList, 'r')
font = cv2.FONT_HERSHEY_SIMPLEX
if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)
for line in tagListFile:
    line = line.strip('\n')
    img_path = line.split(' ')[0]
    tags = line.split(' ')[1:]
    img_name = img_path.split('/')[-1:][0]
    print(img_path)
    #print(img_name)
    img = cv2.imread(img_path)
    for tag in tags:
        x1, y1, x2, y2, class_id = tag.split (",")
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(img, str(class_dict[class_id]), (int(x1),int(y1)), font, 1,(0,0,255),2)
    dets = det_dict[img_name]
    for det in dets:
        class_id, x1, y1, x2, y2, confidence = det
        name = str(class_dict[class_id]) +':'+ str(round(float(confidence),2))
        #print(class_id, x1, y1, x2, y2, confidence)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, str(name), (int(x1), int(y1)), font, 1,(0,255,0),2)
        if class_id in tag_list:
            dest_folder = dest_folder + '/contaminants'
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
    cv2.imwrite(dest_folder + '/' +img_name, img)
    #sys.exit()