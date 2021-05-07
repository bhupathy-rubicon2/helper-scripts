import os

APP_VERSION = "1.0"

#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/home/mcwi9/Rohini/rubicon_sow2/training/map_scripts/'

CAPTURE_DEVICE = "ELP_OV2710"

CPU_EXTENDED_OPERATIONS = "/opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so"

RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources/')
FONTS_DIR = os.path.join(RESOURCES_DIR, 'fonts')

MODELS_DIR = os.path.join(RESOURCES_DIR, 'models')

SSD_FPN_OPENVINO_PATH = os.path.join(MODELS_DIR, 'ssd-fpn-openvino')
SSD_FPN_OPENVINO_CONFIG_PATH = os.path.join(SSD_FPN_OPENVINO_PATH, 'detector.xml')
SSD_FPN_OPENVINO_WEIGHTS_PATH = os.path.join(SSD_FPN_OPENVINO_PATH, 'detector.bin')
SSD_FPN_OPENVINO_INFO_PATH = os.path.join(SSD_FPN_OPENVINO_PATH, 'detector.info')
SSD_FPN_OPENVINO_CLASSES_PATH = os.path.join(SSD_FPN_OPENVINO_PATH, 'z-classes.names')
SSD_FPN_OPENVINO_DIMENSIONS_PATH = os.path.join(SSD_FPN_OPENVINO_PATH, 'dimensions.json')

SSD_FPN_OPENVINO_PATH_CPU = os.path.join(MODELS_DIR, 'ssd-fpn-openvino-cpu')
SSD_FPN_OPENVINO_CONFIG_PATH_CPU = os.path.join(SSD_FPN_OPENVINO_PATH_CPU, 'detector.xml')
SSD_FPN_OPENVINO_WEIGHTS_PATH_CPU = os.path.join(SSD_FPN_OPENVINO_PATH_CPU, 'detector.bin')
SSD_FPN_OPENVINO_INFO_PATH_CPU = os.path.join(SSD_FPN_OPENVINO_PATH_CPU, 'detector.info')
SSD_FPN_OPENVINO_CLASSES_PATH_CPU = os.path.join(SSD_FPN_OPENVINO_PATH_CPU, 'z-classes.names')
SSD_FPN_OPENVINO_DIMENSIONS_PATH_CPU = os.path.join(SSD_FPN_OPENVINO_PATH_CPU, 'dimensions.json')

PARALLEL_IMAGES = 4

FRAMES_PATH = os.path.join(ROOT_DIR, 'detections/frames')
RESULTS_PATH = os.path.join(ROOT_DIR, 'detections/results')

MOCK_INPUT_PATH = os.path.join(ROOT_DIR, 'input')
RUBICON_PATH = "/Rubicon"
UPLOAD_PATH = os.path.join(RUBICON_PATH, 'Uploads')
UPLOAD_IMAGE_PATH = os.path.join(UPLOAD_PATH, 'images')
UPLOAD_GPS_PATH = os.path.join(UPLOAD_PATH, 'gps')
DEBUG_PATH = os.path.join(UPLOAD_PATH, 'var')
DEBUG_PATH_FRAMES = os.path.join(DEBUG_PATH, 'frames')
DEBUG_PATH_DETECTIONS = os.path.join(DEBUG_PATH, 'detections')
DEBUG_PATH_RESULTS = os.path.join(DEBUG_PATH, 'results')

DETECTOR_CLASSES_TARGETS = ['Overflowed']
