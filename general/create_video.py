import cv2
import numpy as np

# choose codec according to format needed 
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video=cv2.VideoWriter('/home/mcw/ML_NAS/Rubicon/SOW2/snow-plow/device-images/latest-data/Images/2021_03_02_1fps.avi', fourcc, 1, (1920,1080))
img_file = open('/home/mcw/ML_NAS/Rubicon/SOW2/snow-plow/device-images/latest-data/Images/2021_03_02_img.txt', 'r')
for line in img_file:
    img_path = line.strip('\n').strip('\r').split(' ')[2]
    img = cv2.imread(img_path)
    video.write(img)

#cv2.destroyAllWindows()
video.release()