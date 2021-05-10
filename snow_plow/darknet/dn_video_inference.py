# To run darknet video inference for snow plow
import sys, os
import pdb
import cv2
import numpy as np
import darknet as dn
import argparse
from time import time

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

def main():
    dn.set_gpu(0)
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = dn.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    width = dn.network_width(network)
    height = dn.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    o_name=input_path.split("/")[-1].split(".")[0]
    output_name= str(o_name+"-"+str(int(time()))) + ".avi"
    output_name= '/home/mcw/ML_NAS/Rubicon/SOW2/snow-plow/result_videos/ML_Training_Results/' + output_name
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    result = cv2.VideoWriter(output_name, fourcc, 30.0, (frame_width, frame_height))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        img_for_detect = dn.make_image(width, height, 3)
        dn.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        detections = dn.detect_image(network, class_names, img_for_detect, thresh=args.thresh)
        index = 0
        scale_w = frame_width / width
        scale_h = frame_height / height
        while(index < len(detections) ):
            class_name = (detections[index][0])
            prob = (detections[index][1])
            print(frame_count, class_name, prob, detections[index][2])
            if(float(prob) > 0.50):
                x_mid = int(detections[index][2][0])
                y_mid = int(detections[index][2][1])
                w = int(detections[index][2][2])
                h = int(detections[index][2][3])
                #print(x_mid, y_mid, w, h)
                x1 = int(x_mid - w/2)
                y1 = int(y_mid - h/2)
                if(w > width-1): w = width-1
                if(h > height-1): h = height-1
                if(x1 < 0): x1 = 0
                if(y1 < 0): y1 = 0
                x2 = x1+w
                y2 = y1+h
                img_x1 = int(x1 * scale_w)
                img_y1 = int(y1 * scale_h)
                img_x2 = int(x2 * scale_w)
                img_y2 = int(y2 * scale_h)
                plow_box = frame[img_y1:img_y2, img_x1:img_x2]
                #print(plow_box.shape)
                plow_box = cv2.cvtColor(plow_box, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(plow_box)
                #plow_box = cv2.cvtColor(plow_box, cv2.COLOR_BGR2RGB)
                #imgray = cv2.cvtColor(plow_box,cv2.COLOR_BGR2GRAY)
                #r, g, b = cv2.split(plow_box)
                #imgray = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
                #works good for white trucks videos
                imgray = cv2.GaussianBlur(h, (71, 71), 0)
                #imgray = cv2.GaussianBlur(imgray, (15, 15), 0)
                #edged = cv2.Canny(imgray, 30, 200)
                ret,thresh = cv2.threshold(imgray,75,255,cv2.THRESH_BINARY_INV)
                #thresh = cv2.adaptiveThreshold(imgray,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                #cv2.imwrite("/home/mcw/ML_NAS/Rubicon/SOW2/snow-plow/result_videos/ML_Training_Results/thresh/" + "thresh_" + str(frame_count) + ".png", thresh)
                contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                print(len(contours))
                big_counter = max(contours, key = cv2.contourArea)
                # determine the most extreme points along the contour
                extLeft = tuple(big_counter[big_counter[:, :, 0].argmin()][0])
                extRight = tuple(big_counter[big_counter[:, :, 0].argmax()][0])
                extTop = tuple(big_counter[big_counter[:, :, 1].argmin()][0])
                extBot = tuple(big_counter[big_counter[:, :, 1].argmax()][0])
                cv2.circle(plow_box, extLeft, 8, (0, 0, 0), -1)
                cv2.circle(plow_box, extRight, 8, (0, 0, 0), -1)
                cv2.circle(plow_box, extTop, 8, (0, 0, 0), -1)
                cv2.circle(plow_box, extBot, 8, (0, 0, 0), -1)
                #cv2.drawContours(plow_box, big_counter, -1, (0, 255, 0), 3)
                cv2.imwrite("/home/mcw/ML_NAS/Rubicon/SOW2/snow-plow/result_videos/ML_Training_Results/contour/" + "contour_" + str(frame_count) + ".png", plow_box)
                cv2.rectangle(frame, (img_x1, img_y1), (img_x2, img_y2), (255,153,51), 2)
                cv2.putText(frame, '[' + str(class_name) + ' : ' + str(prob) +  ']', (int(500), int(450)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
                cv2.putText(frame, 'x1: ' + str(img_x1) , (int(500), int(500)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
                cv2.putText(frame, 'y1: ' + str(img_y1) , (int(500), int(550)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
                cv2.putText(frame, 'x2: ' + str(img_x2) , (int(500), int(600)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
                cv2.putText(frame, 'y2: ' + str(img_y2) , (int(500), int(650)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
            index += 1
        result.write(frame)
        frame_count = frame_count + 1
        #dn.print_detections(detections, args.ext_output)
        #dn.free_image(img_for_detect)
        #cv2.imwrite("frame1.png", frame_resized)

if __name__ == '__main__':
    main()