import logging
import numpy as np
from abc import ABC, abstractmethod

from distance import DistanceModel
from service.bin_classification_service import determine_bin_type
from utils import bifocal_view, non_max_suppression

logger = logging.getLogger(__name__)


class BinDetector(ABC):
    def __init__(self, classes, model_version, estimate_distance=False,
                 dimensions_json=None, capture_device=None, threshold=0.5,
                 box_area_limit=1.0):
        self._category_index = classes
        self.model_version = model_version
        self.estimate_distance = estimate_distance
        self.box_area_limit = box_area_limit
        self.threshold = threshold

        if estimate_distance:
            if dimensions_json is None:
                raise ValueError("Dimensions json must be given if estimate_distance is True")
            self.dist_model = DistanceModel(capture_device, dimensions_json)
        logger.debug("Loaded model")

    def predict_on_image(self, imgs, bifocal=False):
        if imgs is None:
            return []
        if type(imgs) is not list:
            imgs = [imgs]
        height, width = imgs[0].shape[:2]
        images_to_be_submitted = []
        for img in imgs:
            if bifocal:
                img_left, img_right = bifocal_view(img)
                images_to_be_submitted.append(img_left)
                images_to_be_submitted.append(img_right)
            else:
                images_to_be_submitted.append(img)

        raw_detections = self._detect(images_to_be_submitted)

        if bifocal:
            processed_detections = []
            for i in range(1, len(images_to_be_submitted), 2):
                for det in raw_detections[i]:
                    det['box'] = list(det['box'])
                    det['box'][0] += width - height
                    det['box'][2] += width - height

                detections = raw_detections[i - 1] + raw_detections[i]

                indices = non_max_suppression(np.array([det["box"] for det in detections]), overlap_threshold=0.8)
                detections = [detections[index] for index in indices]
                processed_detections.append(detections)
        else:
            processed_detections = raw_detections
        for img, detections in zip(imgs, processed_detections):
            for d in detections:
                if 'Overflowed' in d['class'] or 'Bin' in d['class']:
                    x, y = d['box'][0], d['box'][1]
                    w, h = d['box'][2] - x, d['box'][3] - y

                    bin_type = determine_bin_type(img, bounding_box=[x, y, w, h], rgb_format=True)
                    d['type'] = bin_type

                # compute angle of bins with respect to center of the image
                bin_center = d['box'][0] + (d['box'][2] - d['box'][0]) // 2
                d['angle'] = (bin_center - img.shape[1] // 2) / img.shape[1] * 180

        return processed_detections

    def _detect(self, imgs):
        dets = []
        d_boxes, d_scores, d_classes = self._infer(imgs)
        for i, img in enumerate(imgs):
            height, width = img.shape[0], img.shape[1]
            detections = []

            for box, score, cls in zip(d_boxes[i], d_scores[i], d_classes[i]):
                cls_name = self._category_index[int(cls)]
                det = {
                    'box': (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    'score': float(score),
                    'class': cls_name
                }
                if self.validate_det(det, img):
                    if self.estimate_distance:
                        det['distance'] = self.dist_model.distance_to_obj_by_h(det['box'][3] - det['box'][1], height,
                                                                               cls_name)
                        logger.debug("Distance to object is %.2f" % det['distance'])
                    detections.append(det)

            dets.append(detections)
        return dets

    def validate_det(self, det, img):
        height, width = img.shape[0], img.shape[1]
        logger.debug("Class: %s; score: %.2f" % (det['class'], det['score']))

        b_w, b_h = ((det['box'][2] - det['box'][0]), (det['box'][3] - det['box'][1]))
        area = (b_w * b_h) / (width * height)
        logger.debug("Area is %.2f" % area)
        if area > self.box_area_limit:
            return False

        return True

    @abstractmethod
    def _infer(self, imgs):
        pass
