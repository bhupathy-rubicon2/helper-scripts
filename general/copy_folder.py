import sys
import shutil
import cv2
import glob

ann_dir = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/train/combine_dataset/side-load/img/'

DestImageFolder = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/train/combine_dataset/cleaned_trainset/img/'

for filename in glob.glob(ann_dir + "*.jpg"):
  print(filename, DestImageFolder)
  shutil.move(filename, DestImageFolder)