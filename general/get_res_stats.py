import cv2
import sys
import glob
import os

img_dir = sys.argv[1]
label_dir = sys.argv[2]
#viz_dir = sys.argv[3]

i = 0
print(img_dir)
image_480p = 640*480
image_720p = 1280*720
image_1080p = 1920*1080
pot_cnt = 0
image_resolution_stats = {'less_than_480p' : 0, '480p_to_720p' : 0, '720p_to_1080p' : 0, 'greater_than_1080p' : 0}
for filename in glob.glob(img_dir + "/*.jpg"):

        basename = os.path.basename(filename)
        label_file = basename.replace("jpg", "txt")
        img = cv2.imread(filename)
        
        if(os.path.exists(label_dir+label_file)):
            image_height, image_width = img.shape[0], img.shape[1]
            if image_width*image_height < image_480p:
                image_resolution_stats['less_than_480p'] += 1
            elif image_width*image_height >= image_480p and image_width*image_height <= image_720p:
                image_resolution_stats['480p_to_720p'] += 1
            elif image_width*image_height > image_720p and image_width*image_height <= image_1080p:
                image_resolution_stats['720p_to_1080p'] += 1
            elif image_width*image_height > image_1080p:
                image_resolution_stats['greater_than_1080p'] += 1
print(image_resolution_stats)