import hashlib
import logging

import utils
from settings import CAPTURE_DEVICE, SSD_FPN_OPENVINO_CLASSES_PATH, \
    SSD_FPN_OPENVINO_INFO_PATH, SSD_FPN_OPENVINO_WEIGHTS_PATH, SSD_FPN_OPENVINO_CONFIG_PATH, \
    SSD_FPN_OPENVINO_DIMENSIONS_PATH, SSD_FPN_OPENVINO_CLASSES_PATH_CPU, \
    SSD_FPN_OPENVINO_INFO_PATH_CPU, SSD_FPN_OPENVINO_WEIGHTS_PATH_CPU, SSD_FPN_OPENVINO_CONFIG_PATH_CPU, \
    SSD_FPN_OPENVINO_DIMENSIONS_PATH_CPU, PARALLEL_IMAGES

logger = logging.getLogger(__name__)


class ModelService:

    def __init__(self, model_arch, run_on_cpu=False):
        if model_arch == 'ssd_fpn_openvino':
            from model.ssd_openvino_detector import BinDetectorOpenVino

            if run_on_cpu:
                logger.info("Running on CPU")
                self.classes = utils.parse_classes_file(SSD_FPN_OPENVINO_CLASSES_PATH_CPU)
                model_info = utils.parse_info_file(SSD_FPN_OPENVINO_INFO_PATH_CPU)
                weights_path = SSD_FPN_OPENVINO_WEIGHTS_PATH_CPU
                config_path = SSD_FPN_OPENVINO_CONFIG_PATH_CPU
                dimension_path = SSD_FPN_OPENVINO_DIMENSIONS_PATH_CPU
            else:
                logger.info("running on VPU")
                self.classes = utils.parse_classes_file(SSD_FPN_OPENVINO_CLASSES_PATH)
                model_info = utils.parse_info_file(SSD_FPN_OPENVINO_INFO_PATH)
                weights_path = SSD_FPN_OPENVINO_WEIGHTS_PATH
                config_path = SSD_FPN_OPENVINO_CONFIG_PATH
                dimension_path = SSD_FPN_OPENVINO_DIMENSIONS_PATH

            h = ModelService.get_model_hash(weights_path)
            if 'FRAMEWORK_VERSION' in model_info:
                ModelService.check_framework('openvino', model_info['FRAMEWORK_VERSION'], 'inference_engine')
            if 'SHA1' in model_info:
                ModelService.check_hash(h, model_info['SHA1'])
            self.bifocal = model_info['BIFOCAL']
            self.model = BinDetectorOpenVino(config_path,
                                             weights_path,
                                             num_requests=PARALLEL_IMAGES * 2 if self.bifocal else PARALLEL_IMAGES,
                                             classes=self.classes,
                                             model_version=str(model_info['MODEL_VERSION']),
                                             threshold=model_info['THRESHOLD'],
                                             box_area_limit=model_info['BOX_AREA_LIMIT'],
                                             estimate_distance=True,
                                             dimensions_json=dimension_path,
                                             capture_device=CAPTURE_DEVICE,
                                             resize_h=349, resize_w=349,
                                             run_on_cpu=run_on_cpu)

        else:
            raise ValueError(
                "Invalid model type identifier: " + model_arch +
                "Available formats are 'ssd_fpn_openvino'"
            )
        logger.info(model_info)
        logger.info("Detector SHA1 %s" % h)

    @staticmethod
    def add_object_coordinates(results, locations):
        def is_nan(n):
            try:
                float(n)
                return True
            except ValueError:
                return False

        for result, location in zip(results, locations):
            for r in result:
                if is_nan(location['lat']) or is_nan(location['long']) or is_nan(location['bearing']):
                    lat, long = 0, 0  # we no longer support strings on this field, float only
                else:
                    # adjust bearing of object relative to bearing of the truck
                    bearing_obj = location['bearing'] + r['angle']
                    if bearing_obj > 360:
                        bearing_obj -= 360
                    if bearing_obj < 0:
                        bearing_obj += 360

                    lat, long = utils.calculate_dpos(latitude=location['lat'], longitude=location['long'],
                                                     head=bearing_obj,
                                                     dist=r['distance'])

                r['latitude'], r['longitude'] = lat, long

    # Runs the provided image through the model, and returns an (image, result_dict) tuple
    def apply_model(self, images, locations):
        results = self.model.predict_on_image(images, bifocal=self.bifocal)
        self.add_object_coordinates(results, locations)
        return results

    # check the framework for correct version, if submodule name provided the version is checked for the submodule
    @staticmethod
    def check_framework(framework_name, required_version, submodule=None):
        framework = __import__(framework_name)
        if submodule:
            framework = framework.__dict__[submodule]
        if framework.__version__ != required_version:
            raise ValueError(
                "Invalid framework version for {}.  {} found, but {} is required".format(
                    framework_name, framework.__version__, required_version))

    @staticmethod
    def check_hash(computed_hash, info_hash):
        if computed_hash != info_hash:
            raise ValueError(
                "Invalid model binary. Hash failed checked. Expected {} but {} was found".format(
                    info_hash, computed_hash))

    @staticmethod
    def get_model_hash(model_path):
        BUF_SIZE = 65536

        sha1 = hashlib.sha1()

        with open(model_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha1.update(data)

        return str(sha1.hexdigest())
