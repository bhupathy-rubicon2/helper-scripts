Convert SA annotations to csv format
 - Use convert-csv.py to create csv for 6 keypoints annotations & bbox

Infernce Script for Video
  - Use inference_video_nobbox.py for running video inference with pretrained model for keypoints results
  - This script works only with a working hrnet repo 
    - working setup - cl-14 - /home/mcw/Rohini/HRNet-Human-Pose-Estimation/tools/inference_video_nobbox.py
    - env - /home/mcw/Rohini/snowplow_env/