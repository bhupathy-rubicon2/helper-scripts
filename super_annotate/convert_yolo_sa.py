import superannotate as sa

sa.import_annotation(
    input_dir = 'Z:\\Rubicon\\SOW2\\pothole-retag-data\\RDDC_SA\\val\\',
    output_dir = 'Z:\\Rubicon\\SOW2\\pothole-retag-data\\RDDC_SA\\val_out\\',
    dataset_format = "YOLO", 
    dataset_name =  "pothole_rddc_yolo",
    project_type =  "Vector",
    task = "object_detection"
 )