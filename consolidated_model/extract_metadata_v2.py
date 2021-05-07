import sys
import json
import glob
import csv

ann_dir_list_path = sys.argv[1]
all_classes = []
class_statistics_index = {}
image_resolution_stats = {'image_count' : 0, 'less_than_480p' : 0, '480p_to_720p' : 0, '720p_to_1080p' : 0, 'greater_than_1080p' : 0}

small_crop_size = 32*32
medium_crop_size = 96*96

image_480p = 640*480
image_720p = 1280*720
image_1080p = 1920*1080
image_count = 0

csv_columns = ['name', 'avg_img_dim', 'avg_img_w', 'avg_img_h', 'obj_count', 'small_416', 'medium_416', 'large_416', 
                'avg_bbox_size_416', 'avg_bbox_w_416', 'avg_bbox_h_416', 'small_608', 'medium_608', 'large_608', 
                'avg_bbox_size_608', 'avg_bbox_w_608', 'avg_bbox_h_608']


with open('statistics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(image_resolution_stats)

with open("class_list.txt") as f:
    class_list = [line.rstrip() for line in f]
#print(class_list)

class_id = 0
for class_name in class_list:
    #print(class_name)
    class_statistics = {}
    class_statistics['name'] = class_name
    class_statistics['avg_img_dim'] = 0
    class_statistics['avg_img_w'] = 0
    class_statistics['avg_img_h'] = 0
    class_statistics['obj_count'] = 0
    class_statistics['small_416'] = 0
    class_statistics['medium_416'] = 0
    class_statistics['large_416'] = 0
    class_statistics['avg_bbox_size_416'] = 0
    class_statistics['avg_bbox_w_416'] = 0
    class_statistics['avg_bbox_h_416'] = 0
    class_statistics['small_608'] = 0
    class_statistics['medium_608'] = 0
    class_statistics['large_608'] = 0
    class_statistics['avg_bbox_size_608'] = 0
    class_statistics['avg_bbox_w_608'] = 0
    class_statistics['avg_bbox_h_608'] = 0
    class_statistics_index[class_id] = class_statistics
    #print(class_statistics_index[class_id])
    class_id += 1

#print(class_statistics_index)

with open(ann_dir_list_path) as f:
    ann_dir_list = [line.rstrip() for line in f]

for ann_dir in ann_dir_list:
    for filename in glob.glob(ann_dir + "*.json"):
        with open(filename) as f:
            data = json.load(f)
            image_data = data['images']
            for image in image_data:
                image_width = image['width']
                image_height = image['height']
                image_resolution_stats['image_count'] += 1
                if image_width*image_height < image_480p:
                    image_resolution_stats['less_than_480p'] += 1
                elif image_width*image_height >= image_480p and image_width*image_height <= image_720p:
                    image_resolution_stats['480p_to_720p'] += 1
                elif image_width*image_height > image_720p and image_width*image_height <= image_1080p:
                    image_resolution_stats['720p_to_1080p'] += 1
                elif image_width*image_height > image_1080p:
                    image_resolution_stats['greater_than_1080p'] += 1
                
            annotations = data['annotations'] 
            i = -1
            for cat in data['categories']:
                i += 1
                if cat:
                    if 'subcategory' in cat:
                        if cat['subcategory'] != "null" and cat['subcategory'] is not None :
                            if cat['name'] == 'waste_container':
                                class_name = cat['name']
                            else:
                                class_name = cat['name'] + "__" + cat['subcategory']
                        else:
                            class_name = cat['name']
                    else:
                        class_name = cat['name']

                    annotation = annotations[i]
                    bbox = annotation['bbox']
                    w = bbox['x_max'] - bbox['x_min']
                    h = bbox['y_max'] - bbox['y_min']
                    norm_w = w/image_width
                    norm_h = h/image_height
                    w_416 = norm_w * 416
                    h_416 = norm_h * 416
                    w_608 = norm_w * 608
                    h_608 = norm_h * 608
                        
                    for class_id in class_statistics_index:
                        class_stat = class_statistics_index[class_id]
                        if class_name == class_stat['name']:
                            class_stat['obj_count'] += 1
                            class_stat['avg_img_dim'] += image_width*image_height
                            class_stat['avg_img_w'] += image_width
                            class_stat['avg_img_h'] += image_height

                            if w_416*h_416 <= small_crop_size:
                                class_stat['small_416'] += 1
                            elif w_416*h_416 > small_crop_size and w_416*h_416 <= medium_crop_size:
                                class_stat['medium_416'] += 1
                            elif w_416*h_416 > medium_crop_size:
                                class_stat['large_416'] += 1

                            if w_608*h_608 <= small_crop_size:
                                class_stat['small_608'] += 1
                            elif w_608*h_608 > small_crop_size and w_608*h_608 <= medium_crop_size:
                                class_stat['medium_608'] += 1
                            elif w_608*h_608 > medium_crop_size:
                                class_stat['large_608'] += 1

                            class_stat['avg_bbox_size_416'] += w_416*h_416
                            class_stat['avg_bbox_w_416'] += w_416
                            class_stat['avg_bbox_h_416'] += h_416
                            class_stat['avg_bbox_size_608'] += w_608*h_608
                            class_stat['avg_bbox_w_608'] += w_608
                            class_stat['avg_bbox_h_608'] += h_608

                            break

            

for class_id in class_statistics_index:
    class_stat = class_statistics_index[class_id]
    if class_stat['obj_count'] != 0:
        class_stat['avg_img_dim'] = class_stat['avg_img_dim']/class_stat['obj_count']
        class_stat['avg_img_w'] = class_stat['avg_img_w']/class_stat['obj_count']
        class_stat['avg_img_h'] = class_stat['avg_img_h']/class_stat['obj_count']
        class_stat['avg_bbox_size_416'] = class_stat['avg_bbox_size_416']/class_stat['obj_count']
        class_stat['avg_bbox_w_416'] = class_stat['avg_bbox_w_416']/class_stat['obj_count']
        class_stat['avg_bbox_h_416'] = class_stat['avg_bbox_h_416']/class_stat['obj_count']
        class_stat['avg_bbox_size_608'] = class_stat['avg_bbox_size_608']/class_stat['obj_count']
        class_stat['avg_bbox_w_608'] = class_stat['avg_bbox_w_608']/class_stat['obj_count']
        class_stat['avg_bbox_h_608'] = class_stat['avg_bbox_h_608']/class_stat['obj_count']
        
    #print(class_statistics_index[class_id])

#print(class_statistics_index)

with open('statistics.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_resolution_stats['image_count'],
                         image_resolution_stats['less_than_480p'], 
                         image_resolution_stats['480p_to_720p'],
                         image_resolution_stats['720p_to_1080p'],
                         image_resolution_stats['greater_than_1080p']])

with open('statistics.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_columns)


for class_id in class_statistics_index:
    class_stat =  class_statistics_index[class_id]
    with open('statistics.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([class_stat['name'],
                        int(class_stat['avg_img_dim']),
                        int(class_stat['avg_img_w']),
                        int(class_stat['avg_img_h']), 
                        class_stat['obj_count'], 
                        class_stat['small_416'], 
                        class_stat['medium_416'], 
                        class_stat['large_416'],
                        int(class_stat['avg_bbox_size_416']),
                        int(class_stat['avg_bbox_w_416']),
                        int(class_stat['avg_bbox_h_416']),
                        class_stat['small_608'], 
                        class_stat['medium_608'], 
                        class_stat['large_608'],
                        int(class_stat['avg_bbox_size_608']),
                        int(class_stat['avg_bbox_w_608']),
                        int(class_stat['avg_bbox_h_608'])])
