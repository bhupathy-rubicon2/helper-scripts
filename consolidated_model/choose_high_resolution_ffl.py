import os, sys
import shutil
import cv2
import glob
import json

folderToread = sys.argv[1]
DestFolder = sys.argv[2]

#for ffl - class_list = ['bike', 'bin_pos', 'bus', 'motorbike', 'person', 'truck', 'waste_container']
class_list = ['bike', 'bus', 'person', 'truck']
# mat-cont - class_list = ['bottle_glass', 'bottle_plastic', 'paper_waste_magazine', 'paper_waste_newspaper', 'styrofoam_block']

root_path = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/train/'
json_dir = root_path + folderToread + '/annotations/'
DestImageFolder = DestFolder + '/img/'
DestAnnotFolder = DestFolder + '/annotations/'
#print(DestImageFolder, DestAnnotFolder)
if not os.path.exists(DestImageFolder):
  os.makedirs(DestImageFolder)

if not os.path.exists(DestAnnotFolder):
  os.makedirs(DestAnnotFolder)
count = 0
for filename in glob.glob( json_dir + "*.json"):
  imgname = filename.replace('annotations', 'img').strip('.json')
  cv_img = cv2.imread(imgname)
  height, width, _ = cv_img.shape
  #if height >= 720 and width >= 1280:
  if 1:
    im_name = DestImageFolder + 'ffl_' + imgname.split('/')[-1:][0]
    json_name = DestAnnotFolder + 'ffl_' + filename.split('/')[-1:][0]
    print(filename, height, width)
    is_exists = False
    with open(filename) as f:
      data = json.load(f)
      for idx, cat in enumerate(data['categories']):
        try:
            if cat['name'] == 'waste_container':
              name = cat['name']
              #count = count + 1
            else:
              name = cat['name'] + '_' + cat['subcategory']
        except:
          name = cat['name']
        if name in class_list:
          is_exists = True
      if is_exists:
        if os.path.exists(filename):
          shutil.copy(filename, json_name)
        if os.path.exists(imgname):
          shutil.copy(imgname, im_name)
#print("wc count: ", count)
  