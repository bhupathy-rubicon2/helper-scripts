import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin
from skimage.transform import resize

from model.base import BinDetector
from settings import CPU_EXTENDED_OPERATIONS


class BinDetectorOpenVino(BinDetector):

    def __init__(self, cfg_file, weights_file, classes, model_version, num_requests,
                 threshold=0.8, estimate_distance=False, dimensions_json=None,
                 capture_device=None, box_area_limit=1.0, returns_crops=False,
                 resize_w=349, resize_h=349, keep_aspect_ratio=False, run_on_cpu=False):
        """
        :param cfg_file: path to OpenVino config file (.xml)
        :param weights_file: path to OpenVino weights file (.bin)
        :param classes: classes detected by the model
        :param model_version: identifier - version of the model
        :param num_requests: number of parallel requests
        :param threshold: detection threshold (in 0-1)
        :param estimate_distance: compute estimated distance to the detected object or not
        :param dimensions_json: file containing classes dimensions
        :param capture_device: name of the device used to capture frames if estimate_distance is True.
        Possible choices: oneplus.
        :param box_area_limit: filter boxes that are too big (box area/image area)
        :param returns_crops: crop detection from image
        :param resize_w: inference width
        :param resize_h: inference height
        :param keep_aspect_ratio: keep aspect ration when resizing
        Only resize width will be taken in consideration while the height will be determined
        """
        super().__init__(classes, model_version, estimate_distance, dimensions_json, capture_device,
                         threshold, box_area_limit)

        self.returns_crops = returns_crops

        if run_on_cpu:
            self.__plugin = IEPlugin(device='CPU', plugin_dirs=None)
            self.__plugin.add_cpu_extension(CPU_EXTENDED_OPERATIONS)
        else:
            self.__plugin = IEPlugin(device='MYRIAD', plugin_dirs=None)

        self.__network = IENetwork(model=cfg_file, weights=weights_file)
        self.input_blob = next(iter(self.__network.inputs))
        self.out_blob = next(iter(self.__network.outputs))
        self.exec_net = self.__plugin.load(network=self.__network, num_requests=num_requests)
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.keep_aspect_ratio = keep_aspect_ratio

    def _infer(self, imgs):
        ret_boxes, ret_scores, ret_classes = [], [], []
        for i in range(0, len(imgs)):
            height, width, _ = imgs[i].shape
            if self.keep_aspect_ratio:
                ratio = (height / self.resize_h)
                self.resize_w = width / ratio
            processed_img = resize(imgs[i], output_shape=(self.resize_h, self.resize_w), preserve_range=True)
            processed_img = processed_img.transpose(2, 0, 1)
            processed_img = np.asarray(processed_img)
            processed_img = processed_img.reshape(3, self.resize_h, self.resize_w)

            self.exec_net.start_async(request_id=i, inputs={self.input_blob: np.expand_dims(processed_img, axis=0)})

        for i in range(0, len(imgs)):
            status = self.exec_net.requests[i].wait(6000)

            if status < 0:
                raise RuntimeError('Could not run detection')
            detections = self.exec_net.requests[i].outputs['DetectionOutput']

            indices = np.where(detections[0, 0, :, 2] >= self.threshold)
            boxes = detections[0][0][indices][:, 3:] * np.array([width, height, width, height])
            scores = detections[0][0][indices][:, 2]
            classes = detections[0][0][indices][:, 1]

            ret_boxes.append(boxes)
            ret_scores.append(scores)
            ret_classes.append(classes - 1)

        return ret_boxes, ret_scores, ret_classes

    def __del__(self):
        #del self.exec_net
        del self.__network
        del self.__plugin
