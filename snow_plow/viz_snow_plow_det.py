import os, sys
import shutil
import cv2
import glob
import json
from datetime import datetime

json_dir = sys.argv[1]
dest_folder = sys.argv[2]

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

for filename in glob.glob(json_dir + "*.json"):
    print(filename)
    img_name = filename.split('.json')[0] + '.jpg'
    img = cv2.imread(img_name)
    with open(filename) as f:
      data = json.load(f)
      timestamp = data['utc_timestamp']
      confidence = data['model_artifacts'][0]['confidence']
      class_name = data['model_artifacts'][0]['class_name']
      event_time = data['model_artifacts'][0]['event_time']
      #print(confidence, class_name, event_time)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img, 'Plow Status : ' + str(class_name), (int(300), int(250)), font, 1, (0,0,255), 4)
      cv2.putText(img, 'Confidence : ' + str(confidence), (int(300), int(300)), font, 1, (0,255,0), 4)
      cv2.putText(img, 'Event Time : ' + str(event_time), (int(300), int(350)), font, 1, (0,0,0), 4)
      cv2.putText(img, 'Timestamp : ' + str(timestamp), (int(300), int(400)), font, 1, (0,0,0), 4)
      
      if img is not None:
          cv2.imwrite(dest_folder + '/' + img_name.split('/')[-1], img)
      #exit(0)