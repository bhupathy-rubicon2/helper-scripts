# To randomly select images from json folder for intial trials with data from jayashree
import json 
import glob,random,os
import shutil

#input json folder to select the subset
json_folder = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/small_set/labels/'
#root folder path of image 
root_path = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/small_set/images/'
#destination folder path
dest_image_folder = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/small_set/images/'
dest_labels_folder = '/home/mcw/Rohini/ml_nas/Rubicon/SOW2/data/small_set/labels/'

class_dict = {'bag': 0, 'bin': 0,'Wood':0,'person':0,'Pothole':0,'Car':0,'Motorbike':0,'Bike':0,'Truck':0,'Bus':0,
              'Styrofoam_Block':0,'Clamshell_Food':0,'Bottle':0,'Can':0,'Plastic_Film':0,'Rope':0,
              'Electric_Cord':0,'Paper_waste':0,'Hanger':0,'ball':0,'waste_container':0,'Mailbox':0}
              
filenames = random.sample(os.listdir(json_folder), 271)
img_cnt = 0
for fname in filenames:
    srcpath = os.path.join(json_folder, fname)
    imgpath = os.path.join(root_path, fname.strip('.json'))
    #print(srcpath)
    if os.path.getsize(srcpath) > 0:
        f = open(srcpath) 
    else:
        continue
    data = json.load(f)
    is_visible = False
    for fd in data['images']:
        if fd['image_is_visible']:
            is_visible = fd['image_is_visible']
    if not is_visible:
        continue
    else:
        img_cnt += 1
        #print(data['categories'])
        for subcat in data['categories']:
            #print(subcat['name'])
            obj_cnt = 1
            class_dict[subcat['name']] += 1
    f.close()
    #shutil.copy(srcpath, dest_image_folder)
    #shutil.copy(imgpath, dest_labels_folder)    
print('img_cnt:', img_cnt, '\n', class_dict)