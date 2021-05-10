# To copy images from the list of folders and rename it for potholes audit data
import sys
import shutil
import cv2
import os
import glob

DestImageFolder = sys.argv[1]
root_path = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/Pothhole/Audit-Data/pothole_inferences_mar_no_box/'
folder_list = ['27517', '36111', '36105', '27712', '27703', '36116', '36109', '27713', '27701', '27705',
              '27707', '27704', '36110', '36107', '36106','36117', '36108', '36104']

for folder in folder_list:
  folder_path = root_path + folder + '/'
  for filename in glob.glob(folder_path + "*.jpg"):
    print(filename)
    new_image_name = filename.split('/')[-1]
    new_image_name = folder + '_' + new_image_name
    dest_path = DestImageFolder + '/' + new_image_name
    shutil.copy(filename, dest_path)
  '''line = line.strip('\n')
  img_path = root_path + '/images/' + line
  img_path2 = root_path2 + '/images/' + line
  if os.path.exists(img_path):
    #print(img_path)
    label_path =  img_path.replace('images', 'annotations/xmls').split('.jpg')[0] + '.xml'
    #print(label_path)
    if os.path.exists(label_path):
      print(label_path)
      shutil.copy(img_path, DestImageFolder)
      shutil.copy(label_path, DestLabelsFolder)
  elif os.path.exists(img_path2):
    #print(img_path2)
    label_path2 =  img_path2.replace('images', 'annotations/xmls').split('.jpg')[0] + '.xml'
    if os.path.exists(label_path2):
      print(label_path2)
      shutil.copy(img_path2, DestImageFolder)
      shutil.copy(label_path2, DestLabelsFolder)'''
  #line = line.split(' ')
  #newline = "/DATA/Cadence-DLOE1/Cadence-DLOE1-Backup/Data/IMAGENET/ImageNet-original/ImageNet-Train/" + line[0] 
  #newline = line[0]
  #new_line = newline.split('/')[-1]
  #print(line, DestImageFolder)
  #shutil.copy(line, DestImageFolder)