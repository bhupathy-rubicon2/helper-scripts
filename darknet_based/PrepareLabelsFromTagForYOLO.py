#!/usr/bin/python
# To Create yolo labels from tags format
import glob
import os
import shutil
#import Image
from PIL import Image

imageDir = '/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/train/img_renamed/'
tagDir = '/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/train/new_tag/2class/tags/'
labelsDir = '/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/train/new_tag/2class/labels/'
#imageList = '/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/val/new_tag/2class/val.txt'


#imageListFile = open(imageList, 'w')
imagesprocessed = 0
tags_count = 0
pot_count = 0
mh_count = 0
for image in sorted(glob.glob(imageDir + "*.jpg")):
    imagesprocessed = imagesprocessed + 1
    img = Image.open(image)
    imgW , imgH = img.size
    name = os.path.basename(image)
    tagName = tagDir + name + '.tags'
    if not os.path.exists(tagName):
        continue
    fileId = open(tagName , 'r')
    content = fileId.read().strip('\n').split('\t')
    fileId.close()
    if content[0] != '0' :
        deleteFileflag = True
        #labelFile = open(labelsDir + name.split('.jpg')[0] + '.txt' , "w")
        for tagCount in range(0,int(content[0])):
            x , y , w , h , zero , tag = content[tagCount + 1].strip('\n').split(' ')
            tag = tag.strip('\n')
            #if tag != '0' :
                #print 'Ignore case' , image , tag
            #else :
            #if tag in ('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'):
            if tag in ('101', '102'):
                deleteFileflag = False
                print('processing :', image,'Tag: ',labelsDir + name.split('.jpg')[0] + '.txt' , tag)
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                Cx = ( ( x + ( ( w * 1.) / 2 ) ) * 1. ) / imgW
                Cy = ( ( y + ( ( h * 1.) / 2 ) ) * 1.) / imgH
                iW = ( w * 1.) / imgW
                iH = ( h * 1.) / imgH
                tags_count = tags_count + 1
                if tag == '101':
                    cid = 0
                    pot_count = pot_count + 1
                elif tag == '102':
                    cid = 1
                    mh_count = mh_count + 1
        #        labelFile.write(str(cid)+ " "+ str(Cx) + " " + str(Cy) + " " + str(iW) + " " + str(iH) + '\n')
        #labelFile.close()
        if deleteFileflag:
            os.remove(labelsDir + name.split('.jpg')[0] + '.txt')
        #else:
        #    imageListFile.write(image+'\n')
    #else:
        #print 'removing image' , image , 'coz its ignore case'
#imageListFile.close()
print("Tags Count: ", pot_count, mh_count)
print('Total Images Processed', imagesprocessed)
