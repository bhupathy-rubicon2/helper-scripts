# Dump detections from yolo csv to rubicon format for viz or map calculation
import os, sys
import shutil
import argparse
import csv

class_ids = {'0':'bag_paper',
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
           
def LoadCSVToList(csv_file):
  with open(csv_file) as csv_file_object:
    csv_reader_object = csv.reader(csv_file_object)
    return list(csv_reader_object)

det_folder = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/data/train/combine_dataset/testset/detection/detections'
try:
  shutil.rmtree(det_folder)
except:
  print("Folder doesn't Exist:", det_folder)
os.mkdir(det_folder)
detections_all = LoadCSVToList('./darknet/detections_y10_55k_0.2t.csv')
for det in detections_all:
  img_name = det[1]
  det_file = img_name.split('.jpg')[0] + '.txt'
  f = open(det_folder + '/'+det_file, 'a')
  c_name = class_ids[det[2]]
  prob = det[7]
  x1, y1, x2, y2 = det[3], det[4], det[5], det[6]
  f.write(c_name+' '+prob+' '+x1+' '+y1+' '+x2+' '+y2+'\n')