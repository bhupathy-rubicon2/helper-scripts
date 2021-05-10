# Copyright (c) 2016 Artsiom Sanakoyeu
# Script to convert annotated snow plow keypoints to csv format
from __future__ import division
from __future__ import print_function

from os.path import basename
from scipy.io import loadmat
import argparse
import glob
import re
import os.path
import json
import csv
import random

def create_data(images_dir, transpose_order=(2, 0, 1)):
    """
    Create a list of lines in format:
      image_path, x1, y1, x2,y2, ...
      where xi, yi - coordinates of the i-th joint
    """
    files=[]
    for ext in ('*.png', '*.jpg'):
        files.extend(glob.glob(os.path.join(images_dir, ext)))

        # for img_path in sorted(glob.glob(os.path.join(images_dir, '*.png'))):
    # print(files)
    key_point_all=[]
    lines = list()
    for img_path in sorted(files):
        
        annotation=img_path+".json"
        print(annotation)

        if(os.path.isfile(annotation) ):
            with open(annotation) as json_file:
                data = json.load(json_file)
                key_point=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                pose=[]
                #print(pts)
                instances = data['instances']
                for instance in instances:
                    if instance['type'] == "bbox":
                        points = instance['points']
                        key_point[13]=int(points["x1"])
                        key_point[14]=int(points["y1"])
                        key_point[15]=int(points["x2"])
                        key_point[16]=int(points["y2"])
                for pts in data["instances"]:
                    if pts["classId"]==2:
                        key_point[1]=int(pts["x"])
                        key_point[2]=int(pts["y"])
                    if pts["classId"]==6:
                        key_point[3]=int(pts["x"])
                        key_point[4]=int(pts["y"])
                    if pts["classId"]==3:
                        key_point[5]=int(pts["x"])
                        key_point[6]=int(pts["y"])
                    if pts["classId"]==5:
                        key_point[7]=int(pts["x"])
                        key_point[8]=int(pts["y"])
                    if pts["classId"]==4:
                        key_point[9]=int(pts["x"])
                        key_point[10]=int(pts["y"])
                    if pts["classId"]==7:
                        key_point[11]=int(pts["x"])
                        key_point[12]=int(pts["y"])
                #if pts["classId"]==7:
                #    try:
                #    except:
                #        continue
                # pose=[pts["x"],pts["y"]]
                key_point[0]=str(img_path.split("/")[-1])
                print(key_point)
                key_point_all.append(key_point)
            # exit(1)
        # continue
        # index = int(re.search(r'im([0-9]+)', basename(img_path)).groups()[0]) - 1
        # joints_str_list = [str(j) if j > 0 else '-1' for j in joints[index].flatten().tolist()]

        # out_list = [os.path.basename(img_path)]
        # out_list.extend(joints_str_list)
        # out_str = ','.join(out_list)

        # lines.append(out_str)
    return key_point_all


if __name__ == '__main__':
    """
    Write train.csv and test.csv.
    Each line in csv file will be in the following format:
      image_name, x1, y1, x2,y2, ...
      where xi, yi - coordinates of the i-th joint
    Train file consists of 11000 lines (all images from extended LSP + first 1000 images from small LSP).
    Test file consists of 1000 lines (last 1000 images from small LSP).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--small_lsp_images_dir', type=str, default=os.path.join("."))
    parser.add_argument('--output_dir', type=str, default="../")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_train = open('%s/train_joints_rohini.csv' % args.output_dir, 'w')
    file_test = open('%s/test_joints_rohini.csv' % args.output_dir, 'w')


    print('Read LSP')
    key_point = create_data(args.small_lsp_images_dir,
                                  transpose_order=(2, 1, 0))  # different dim order
    print('Small LSP images:', len(key_point))

    num_small_lsp_train = 20

    #random.shuffle(key_point) 

    for line in key_point[:num_small_lsp_train]:
        print(line, file=file_train)
    for line in key_point[num_small_lsp_train:]:
        print(line, file=file_test)


