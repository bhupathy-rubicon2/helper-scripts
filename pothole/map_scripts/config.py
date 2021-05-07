import os

#ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))

#DETECTION_DATA_DIR = os.path.join('/home/mcwi9/Model_Optimization/data/', 'detection-v2')
#DETECTION_VAL_DIR = os.path.join('/home/mcwi9/Model_Optimization/data/', 'validation-v2')
#TEST_DIR = os.path.join('/home/mcwi9/Model_Optimization/data/', 'test-v2')

#VAL_SPLIT = 0.15
#AUG_DEGREE = 6
#RANDOM_SEED = 42

#PROCESSED_DIR = os.path.join('/home/mcwi9/Model_Optimization/darknet/test_data/data', 'processed')
#PROCESSED_DETECTION_DIR = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/pothole-retag-data/trainings/model_results/Rub-tag-R1-RDDC/'
#PROCESSED_DETECTION_DIR = os.path.join('/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/trainings/cl-29', 'detection')
#PROCESSED_MXNET_DETECTION_DIR = os.path.join('PROCESSED_DETECTION_DIR', 'mxnet')
#PROCESSED_CROPS_DIR = os.path.join(PROCESSED_DIR, 'crops')
#PROCESSED_PREFILTER_DIR = os.path.join(PROCESSED_DIR, 'prefilter')

MODELS_DIR = os.path.join('/mnt/i/Rohini/Rubicon/resources/models')
#FILTER_MODELS_DIR = os.path.join('models', 'filter')

#RESOURCES_DIR = os.path.join("resources")

#BIN_DETECTIONS = "/tmp/{}-detection.txt"
#DETECTION_CLASSES_FILE = os.path.join("object-detection.pbtxt")
#Z_CLASSES_FILE = os.path.join("z-classes.names")
#Z_DISTANCE_FILE = os.path.join("dimensions.json")
ROOT_DIR = "/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/pothole-retag-data/model_results/Rub-T11e/"
ITER = "6k_audit_data_09t/"
CONF_SCORE = "0.9"
TEST_GROUND_TRUTH_PATH = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/Pothhole/Audit-Data/audit_data_tagged/ground-truth/'
TEST_DETECTIONS_PATH = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/trainings/cl-12/yolo_backup/detections/'
TEST_IMAGES_PATH = os.path.join('/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/Pothhole/Audit-Data/audit_data_tagged/', 'labels')
TEST_GT_IMAGES_PATH = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/Pothhole/Audit-Data/audit_data_tagged/gt_viz/'
#AUG_DIR = "/tmp/aug_dir"


#CACHE_DIR = "/tmp/cache"
#VALIDATION_CSV = os.path.join(PROCESSED_DETECTION_DIR, "val-labels.csv")

#ANNO_PATH = os.path.join(DETECTION_DATA_DIR, "annotations")
#ANNO_PATH_YOLO = os.path.join(DETECTION_DATA_DIR, "labels")
#IMG_PATH = os.path.join(DETECTION_DATA_DIR, "img")

#CLASSES_FILE = os.path.join(PROCESSED_DETECTION_DIR, "residential.names")
#DATA_DESCR_FILE = os.path.join(PROCESSED_DETECTION_DIR, "residential.data")

#DEBUG = False

#CPU_EXTENDED_OPERATIONS = "/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so"

'''TRAFFIC_SIGNS = ["traffic_sign_stop", "traffic_sign_yield", "traffic_sign_dead_end",
                 "traffic_sign_crosswalk", "traffic_sign_stop_here_on_red",
                 "traffic_sign_speed_limit_ahead", "traffic_sign_cross_road",

                 "traffic_sign_speed_limit_5", "traffic_sign_speed_limit_10",
                 "traffic_sign_speed_limit_15", "traffic_sign_speed_limit_20",
                 "traffic_sign_speed_limit_25", "traffic_sign_speed_limit_30",
                 "traffic_sign_speed_limit_35", "traffic_sign_speed_limit_40",
                 "traffic_sign_speed_limit_45", "traffic_sign_speed_limit_50",
                 "traffic_sign_speed_limit_55", "traffic_sign_speed_limit_60",
                 "traffic_sign_speed_limit_65", "traffic_sign_speed_limit_70",
                 "traffic_sign_speed_limit_25_ahead",
                 "traffic_sign_stop_ahead", "traffic_sign_two_directions", "traffic_sign_school_crosswalk",
                 "traffic_sign_side_road", "traffic_sign_one_way", "traffic_sign_right_curve_ahead",
                 "traffic_sign_road_work_ahead", "traffic_sign_one_direction_right_arrow",
                 "traffic_sign_one_direction_left_arrow", "traffic_sign_caution_children",
                 "traffic_sign_no_trucks", "traffic_sign_no_left_turn",
                 "traffic_sign_no_parking", "traffic_sign_utility_work_ahead",
                 "traffic_sign_senior_zone", "traffic_sign_traffic_lights",

                 "traffic_sign_slow_20", "traffic_sign_do_not_enter", "traffic_sign_trucks_entering",
                 "traffic_sign_slow", "traffic_sign_speed_hump_ahead", "traffic_sign_trucks_entering_highway",
                 "traffic_sign_turn_right", "traffic_sign_turn_left",
                 "traffic_sign_bike_route", 'traffic_sign_left_or_straight_ahead',
                 "traffic_sign_watch_for_bikes", "traffic_sign_hospital", "traffic_sign_narrow_bridge",
                 "traffic_sign_rough_road", "traffic_sign_no_right_turn",
                 "traffic_sign_school_zone", "traffic_sign_school_zone_ends",
                 "traffic_sign_reverse_turn_right", "traffic_sign_reverse_turn_left",
                 "traffic_sign_round_about", "traffic_sign_left_curve_ahead",
                 "traffic_sign_right_only", "traffic_sign_left_only", "traffic_sign_straight_only",
                 "traffic_sign_children_zone",
                 "traffic_sign_rail_road", "traffic_sign_ahead_only", "traffic_sign_yield_ahead",
                 "traffic_sign_do_not_pass",
                 "traffic_sign_dip", "traffic_sign_church_zone", "traffic_sign_bump",
                 "traffic_sign_right_lane_must_turn_right", "traffic_sign_left_lane_must_turn_left",
                 "traffic_sign_drugfree_zone", "traffic_sign_arrow_right",
                 "traffic_sign_narrow_road", "traffic_sign_limited_sight", "traffic_sign_no_turn_on_red",
                 "traffic_sign_bridge_height", "traffic_sign_deer", "traffic_sign_hairpin_curve",
                 "traffic_sign_share_the_road",
                 "traffic_sign_reduced_speed_ahead", "traffic_sign_blind_persons",
                 "traffic_sign_no_outlet", "traffic_sign_zigzag_road",
                 "traffic_sign_blind_driveway", "traffic_sign_school_zone_begins",
                 "traffic_sign_speed_humps_ahead", "traffic_sign_right_turn_ahead",
                 "traffic_sign_ramp_metered_when_flashing",
                 "traffic_sign_trucks_prohibited", "traffic_sign_arrow",
                 "traffic_sign_bikes", "traffic_sign_right_on_green_only",
                 "traffic_sign_limited_sight_distance",
                 "traffic_sign_lane_ends_merge_left", "traffic_sign_road_work",
                 "traffic_sign_right_or_straight_ahead", "traffic_sign_straight_or_turn_right",
                 "traffic_sign_road_work_500_ft", "traffic_sign_hill_blocks_view",
                 "traffic_sign_intersection_ahead", "traffic_sign_road_work"
                                                    "traffic_sign_y_intersection", "traffic_sign_no_turn_left",
                 "traffic_sign_no_turn_right",
                 "traffic_sign_may_use_full_lane", "traffic_sign_bikes_may_use_full_lane",
                 "traffic_sign_end_road_work", "traffic_sign_detour_arrow_right",
                 "traffic_sign_detour", "traffic_sign_center_lane_only",
                 "traffic_sign_right_lane_closed", "traffic_sign_arrow_road_left",
                 "traffic_sign_right_lane_ends", "traffic_sign_deaf_child_at_play",
                 "traffic_sign_no_turns", "traffic_sing_divided_highway_ends",
                 "traffic_sign_turn_right_or_may_continue_straight", "traffic_sign_right",
                 "traffic_sign_left", "traffic_sign_arrow_left", "traffic_sing_two_way",
                 "traffic_sign_road_curve", "traffic_sign_speed_cushions_ahead",
                 "traffic_sign_do_not_block_intersection", "traffic_sign_watch_for_pedestrians"]

CLASSES_EXTENSIONS = {"waste_container": ["waste_container_wm", "waste_container_republic", "waste_container_aw", "waste_container_other",
                      "waste_container_wca", "waste_container_ctr", "waste_container_advanced_disposal",
                      "waste_container_iesi", "waste_container_progressive", "waste_container_waste_connections"],
                      "car": ["car_white", "car_silver", "car_black", "car_grey", "car_blue",
                              "car_red", "car_brown", "car_green", "car_other"],
                      "bin_pos": ["bin_pos_green", "bin_pos_blue", "bin_pos_black", "bin_pos_general"],
                      "bin_neg": ["bin_neg_green", "bin_neg_blue", "bin_neg_black", "bin_neg_general"]}

CLASSES_CORRESP = dict((v,k) for k in CLASSES_EXTENSIONS for v in CLASSES_EXTENSIONS[k])

AGGREGATE_CLASSES = True

OBJECTS = ["truck", "bus", "motorbike", "person", "shrub",
           "bike", "pothole", "license_plate", "dog", "incorrect_object"]

if AGGREGATE_CLASSES:
    CLASSES = ["trash_object", "bin_other", "bin_other_pos"] + OBJECTS + TRAFFIC_SIGNS + list(CLASSES_EXTENSIONS.keys())
else:
    CLASSES = ["trash_object", "bin_other", "bin_other_pos"] + OBJECTS + TRAFFIC_SIGNS + list(CLASSES_CORRESP.keys())'''
