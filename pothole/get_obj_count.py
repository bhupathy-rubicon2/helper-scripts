# Script to get object count from yolo labels
import cv2
import sys
import glob
import os

img_dir = sys.argv[1]
label_dir = sys.argv[2]
#viz_dir = sys.argv[3]

i = 0
print(img_dir)
#print(glob.glob(img_dir + "*.jpg"))
pot_cnt = 0
for filename in glob.glob(img_dir + "/*.jpg"):

        basename = os.path.basename(filename)
        label_file = basename.replace("jpg", "txt")
        img = cv2.imread(filename)
        
        if(os.path.exists(label_dir+label_file)):
                fl = open(label_dir + label_file, 'r')
                data = fl.readlines()
                fl.close()

                for dt in data:

                # Split string to float
                        cls_name, x1, y1, x2, y2 = dt.split(' ')
                        #prob = str(round(float(prob), 2))
                        #print(cls_name, prob, x1, y1, x2, y2)
                        #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                      
                        if cls_name == "0":
                            pot_cnt = pot_cnt + 1
                            #cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                            #img = cv2.putText(img, prob, (x1, y2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
                        #else:
                        #    cv2.rectangle(img, (l, t), (r, b), (255, 0, 255), 2)
                        #    img = cv2.putText(img, 'GT-man', (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


        #cv2.imwrite(viz_dir + basename, img)
        #print("file : ", basename)
        print("count : ", i)
        i += 1
print("pot count : ", pot_cnt)
