# Script to dump detections from yolo detections csv
import os, sys
import shutil
import argparse
import csv

import sys

det_file = sys.argv[1]
out_dir = sys.argv[2]

class_ids = {'0':'pothole', 
             '1':'manhole'}
           
def LoadCSVToList(csv_file):
  with open(csv_file) as csv_file_object:
    csv_reader_object = csv.reader(csv_file_object)
    return list(csv_reader_object)

det_folder = out_dir


try:
  shutil.rmtree(det_folder)
except:
  print("Folder doesn't Exist:", det_folder)
os.mkdir(det_folder)
detections_all = LoadCSVToList(det_file)
for det in detections_all:
  img_name = det[1]
  det_file = img_name.split('.jpg')[0] + '.txt'
  f = open(det_folder + '/'+det_file, 'a')
  c_name = class_ids[det[2]]
  prob = det[7]
  x1, y1, x2, y2 = det[3], det[4], det[5], det[6]
  f.write(c_name+' '+prob+' '+x1+' '+y1+' '+x2+' '+y2+'\n')
