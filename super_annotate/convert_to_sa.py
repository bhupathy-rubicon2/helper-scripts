import superannotate as sa

'''sa.import_annotation(
    "/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/check_sa/img",
    "/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/check_sa/output/",
    "VOC", "pothole_voc", "Vector", "object_detection"
 )'''

sa.import_annotation(
    "/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/RDDC_SA/train",
    "/home/mcw/ML_NAS/Rubicon/SOW2/pothole-retag-data/RDDC_SA/train_out/",
    "VOC", "pothole_voc", "Vector", "object_detection"
 )