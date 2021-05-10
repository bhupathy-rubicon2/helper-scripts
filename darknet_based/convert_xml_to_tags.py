# To Convert ground truth xml files to tags format
import xml.etree.ElementTree as ET
import os
import sys
import cv2

inputFolderName = '/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/val/new_tag/audit-data_xml/'
outputPath = '/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/val/new_tag/2class/tags/'
imageDir = '/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/val/audit-data/img/'
modimageDir = '/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/val/img_renamed/'

#new tagging lt project
classes = {'pothole': 101,
           'manhole': 102}
'''
           'bin_other': 1,
           'bin_pos': 2,
           'bus': 3,
           'car': 4,
           'person': 5,
           'trash_object': 6,
           'truck': 7,
           'waste_container': 8,
           'traffic_sign_crosswalk': 9,
           'traffic_sign_school_crosswalk': 10,
           'traffic_sign_speed_limit_25': 11,
           'traffic_sign_speed_limit_35': 12,
           'traffic_sign_stop': 13,
           'traffic_sign_stop_ahead': 14,
           }'''
#res_file = open('img_res.txt', 'w')
for path, subdirs, files in os.walk(inputFolderName):
        if len(files) > 0:
            for dirFile in files:
                #print "dirfile " + dirFile
                filename = os.path.join(path,dirFile)
                if os.stat(os.path.join(path,dirFile)) == 0:
                    continue
                print (" filename " + filename)
                if filename.endswith(".xml"):
                    outputFilename = dirFile.replace('.xml','')
                    #print "outputfilename " + outputFilename
                    tag_count = 0
                    Taglist = []
                    tree = ET.parse(filename)
                    root = tree.getroot()
                    for LabelingTool in root.iter('LabelingTool'):
                      creator = LabelingTool.find('creator').text # creator
                      version = LabelingTool.get('version') # version
                      for project in LabelingTool.findall('project'):
                          name = project.get('name') # project_name
                          for clas in project.findall('class'):
                              class_id = clas.get('id')
                              for box in clas.findall('boundingbox'):  # B_box values
                                invisible = "False"
                                atrList = None
                                attr = None
                                x1 = box.find('x1').text
                                x2 = box.find('x2').text
                                y1 = box.find('y1').text
                                y2 = box.find('y2').text
                                angle = 0
                                w = str(int(x2) - int(x1)) # width
                                h = str(int(y2) - int(y1)) # height # can be used if needed 
                                if class_id == "101" or class_id == "102":
                                  tag_count = tag_count + 1
                                  Taglist.append((x1,y1,w,h,angle,int(class_id)))
                                  #print( x1 , y1 , w , h , angle , classes[cls])
                print(imageDir + outputFilename )
                if os.path.exists(imageDir + outputFilename):
                  img = cv2.imread(imageDir + outputFilename )
                  new_name = modimageDir + "ad_" + outputFilename 
                  cv2.imwrite(new_name, img)
                if tag_count > 0:
                  fileId = open(outputPath+ "ad_" + outputFilename+".tags", "w+")
                  fileId.write(str(tag_count))
                  for tag in Taglist:
                    x1,y1,w,h,angle,class_id = tag
                    fileId.write('\t' + str(x1) +" "+str(y1)+" "+str(w)+" "+str(h)+" "+str(angle)+" "+str(class_id))
                  fileId.close()

