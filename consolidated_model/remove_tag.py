import os, sys
import json
import shutil
import cv2
import glob

root_path = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/train/front-facing-lateral/selected/'
dest_path = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/train/front-facing-lateral/selected_blacked/'
json_dir = root_path + '/annotations/'
img_dir = root_path + '/img/'

if not os.path.exists(dest_path + '/annotations/'):
  os.makedirs(dest_path + '/annotations/')

if not os.path.exists(dest_path + '/img/'):
  os.makedirs(dest_path + '/img/')

class_names = ['mailbox']
# for mt-cont - class_names = ['bag_paper', 'bag_plastic', 'bin_neg', 'bin_pos', 'can', 'person', 'plastic_film']
# for ad - class_names = ['bag_paper', 'bag_plastic', 'bike', 'bin_pos', 'motorbike']

with open("class_list.txt") as f:
    class_list = [line.rstrip() for line in f]
#print(class_list)
#sys.exit()
for filename in glob.glob( json_dir + "*.json"):
  imgname = filename.replace('annotations', 'img').strip('.json')
  cv_img = cv2.imread(imgname)
  # chech if its 'rl_'
  with open(filename) as f:
    data = json.load(f)
    annotations = data['annotations']
    remove_list = []
    for elem in class_names:
      for idx, cat in enumerate(data['categories']):
        if cat:
          try:
            if cat['subcategory'] != "null" and cat['subcategory'] is not None :
              name = cat['name'] + '_' + cat['subcategory']
          except:
            name = cat['name']
          if name in class_names:
            print(filename)
            remove_list.append(cat['id'])
            for key in list(cat):
                del cat[key]
    #print(annotations)
    for annot in annotations:
      if annot['id'] in remove_list:
        y_min = annot['bbox']['y_min']
        y_max = annot['bbox']['y_max']
        x_min = annot['bbox']['x_min']
        x_max = annot['bbox']['x_max']
        cv2.rectangle(cv_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), -1)
        for key in list(annot):
          del annot[key]
    #print(data)
    new_jname = filename.replace(root_path, dest_path)
    new_iname = imgname.replace(root_path, dest_path)
    #print(filename, imgname)
    #print(new_jname, new_iname)
    data_file = open(new_jname, 'w')
    json.dump(data, data_file)
    if cv_img is not None:
      cv2.imwrite(new_iname, cv_img)
  #sys.exit()