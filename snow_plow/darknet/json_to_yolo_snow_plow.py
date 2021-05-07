import json
import glob
import sys
import cv2

input_dir = sys.argv[1]
labelsDir = "/home/mcw/ML_NAS/Rubicon/SOW2/snow-plow/mcw_tagged_data/labels/"
image_list = "/home/mcw/ML_NAS/Rubicon/SOW2/snow-plow/mcw_tagged_data/snow_plow_img.txt"
imageListFile = open(image_list, "w")
print(input_dir)
for f in glob.glob(input_dir + "/*.json"):
    with open(f,) as infile:
        data = json.load(infile)
        #print(data)
        metadata = data['metadata']
        filename = metadata['name']
        print("filename : ", filename)
        instances = data['instances']
        img = cv2.imread(input_dir+filename)
        img_h, img_w, channels = img.shape
        if '.png' in filename:
            labelFile = open(labelsDir + filename.split('.png')[0] + '.txt' , "w")
        elif '.jpg' in filename:
            labelFile = open(labelsDir + filename.split('.jpg')[0] + '.txt' , "w")
        for instance in instances:
            if instance['type'] == "bbox":
                #class id is 0 for plow_box
                tag = 0
                points = instance['points']
                x1,x2,y1,y2 = points['x1'], points['x2'], points['y1'], points['y2'] 
                #print("Box coords : ", x1, x2, y1, y2)
                w = x2-x1
                h = y2-y1
                #center_x = int(x1 + w/2)
                #center_y = int(y1 + h/2)
                Cx = ( ( x1 + ( ( w * 1.) / 2 ) ) * 1. ) / img_w
                Cy = ( ( y1 + ( ( h * 1.) / 2 ) ) * 1.) / img_h
                iW = ( w * 1.) / img_w
                iH = ( h * 1.) / img_h
                #print("center_x, center_y, w, h : ", center_x, center_y, w, h)
                labelFile.write(str(tag)+ " "+ str(Cx) + " " + str(Cy) + " " + str(iW) + " " + str(iH) + '\n')
                print("Normalized center_x, center_y, w, h : ", Cx, Cy, iW, iH)
        labelFile.close()
        imageListFile.write(input_dir+filename+'\n')
        #exit(1)