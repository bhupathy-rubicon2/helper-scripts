## Script to upload images and annotations to SA
## path to token config with structure {"token": "303f4adb15d1373c1bb0ccb7"}
## token_path = "./config.json"
## sa.init(token_path)

import superannotate as sa

sa.upload_images_from_folder_to_project(
        project = "Pothole-daily-data-tagging/2021_04_08",
        folder_path = 'Z:\\Rubicon\\SOW2\\Pothhole\\rear-load-dataset-front-facing\\2021_04_08')
        
'''sa.upload_images_from_folder_to_project(
        project = "Pothhole-retag-viz/Review-RDDC-Val-Clean",
        folder_path = 'Z:\\Rubicon\\SOW2\\pothole-retag-data\\RDDC_SA\\val_out_cleaned')'''

'''sa.upload_annotations_from_folder_to_project(
        project = "Pothhole-retag-viz/Review-RDDC-Val-Clean", 
        folder_path = 'Z:\\Rubicon\\SOW2\\pothole-retag-data\\RDDC_SA\\val_out_cleaned')'''