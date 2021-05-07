import os, sys
import shutil
import cv2
import glob

folderToread = sys.argv[1]
DestFolder = sys.argv[2]

root_path = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/train/'
json_dir = root_path + folderToread + '/annotations/'
DestImageFolder = DestFolder + '/img/'
DestAnnotFolder = DestFolder + '/annotations/'
#print(DestImageFolder, DestAnnotFolder)
if not os.path.exists(DestImageFolder):
  os.makedirs(DestImageFolder)

if not os.path.exists(DestAnnotFolder):
  os.makedirs(DestAnnotFolder)

for filename in glob.glob( json_dir + "*.json"):
  imgname = filename.replace('annotations', 'img').strip('.json')
  cv_img = cv2.imread(imgname)
  height, width, _ = cv_img.shape
  if height >= 720 and width >= 720:
    im_name = DestImageFolder + 'rl_' + imgname.split('/')[-1:][0]
    json_name = DestAnnotFolder + 'rl_' + filename.split('/')[-1:][0]
    print(filename, height, width)
    if os.path.exists(filename):
      shutil.copy(filename, json_name)
    if os.path.exists(imgname):
      shutil.copy(imgname, im_name)
  