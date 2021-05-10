## To convert yolo annotations to pascal voc xml format
import xml.etree.ElementTree as ET
import os
import sys
import cv2
from pascal_voc_writer import Writer

labelsList = '/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/RDDC_SA/train.txt'
fileptr = open(labelsList, 'r')

for line in fileptr:
    line = line.strip('\n')
    img_path = line.replace('Annotations','Images').replace('.txt', '.jpg')
    print(line)
    img = cv2.imread(img_path)
    img_name = img_path.split('/')[-1]
    #print(img_name)
    height, width, channels = img.shape[0], img.shape[1], img.shape[2]
    writer = Writer(img_name, width, height)
    #print(height, width, channels)
    tag_file = open(line, 'r')
    for tags in tag_file:
        print(tags)
        tag = tags.strip('\n').strip('\r')
        elms = tag.split(' ')
        #print(tag, elms)
        xmin = max(float(elms[1]) - float(elms[3]) / 2, 0)
        xmax = min(float(elms[1]) + float(elms[3]) / 2, 1)
        ymin = max(float(elms[2]) - float(elms[4]) / 2, 0)
        ymax = min(float(elms[2]) + float(elms[4]) / 2, 1)
        xmin = int(width * xmin)
        xmax = int(width * xmax)
        ymin = int(height * ymin)
        ymax = int(height * ymax)
        writer.addObject('pothole', xmin, ymin, xmax, ymax)
    xml_name = img_name.split('.jpg')[0] + '.xml'
    xml_path = "/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/RDDC_SA/train/Xml/" + xml_name
    writer.save(xml_path)
    #exit(0)
