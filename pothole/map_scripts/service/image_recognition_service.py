import time

import cv2
import datetime
import json
import logging
import numpy as np
import os
from rubi_bus_client import Channels, subscribe_camera_feed
from threading import Lock, Thread

from brisk_filter import BriskFilter
from settings import APP_VERSION, FRAMES_PATH, RESULTS_PATH, UPLOAD_IMAGE_PATH, DEBUG_PATH_FRAMES, \
    DEBUG_PATH_DETECTIONS, DEBUG_PATH_RESULTS, DETECTOR_CLASSES_TARGETS, PARALLEL_IMAGES
from utils import dhash, hash_dist, write_bb, encode_b64, extract_data_from_json

logger = logging.getLogger(__name__)


#
# Define global handler for serialization of custom class to json
#

def jdefault(o):
    return o.__dict__


class TrashDetectionModel:

    def __init__(self):
        self.ApplicationVersion = APP_VERSION
        self.UtcTimestamp = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
        self.Longitude = ""
        self.Latitude = ""
        self.ModelVersion = ""
        self.DeviceId = os.environ.get("RUBICON_DEVICE_ID", "unassigned_device")
        self.DeviceGeneratedName = time.strftime("%Y_%m_%d_%H_%M_%S")
        self.Image = None
        self.ModelArtifacts = None
        self.Bearing = None
        self.Speed = None
        self.RawBearing = None
        self.ModelClass = "BinOverflowDetection"
        self.IsValidGps = None


class ImageRecognitionService:
    def __init__(self, waiter, model, img_quality, rescale_factor,
                 filter_duplicates=True, hash_threshold=30, timing=False, save_all=False, blur_on=True,
                 debug_mode=False, detect_all=False, detect_bin_and_overflow_bin=False, resolution='1080p',
                 bus_service_mock=None):
        """

        :param waiter: wait policy
        :param model: model service
        :param img_quality:
        :param rescale_factor: rescale factor for results
        :param filter_duplicates: filter consecutive frames that are similar
        :param hash_threshold: duplicate filtering threshold
        :param timing: log timings
        :param save_all: save all frames, if False will save only detections
        :param blur_on: blur everything that it's outside detection
        :param debug_mode: save data to debug path
        :param detect_all: detect and save all classes
        :param detect_bin_and_overflow_bin: detect and save both bin and overflowing bin
        :param resolution: image resolution
        :param bus_service_mock
        """
        self.wait_policy = waiter
        self.model_service = model
        self.img_quality = img_quality
        self.rescale_factor = rescale_factor
        self.filter_duplicates = filter_duplicates
        self.ref_hash = None
        self.ref_image = None
        self.ref_boxes = None
        self.ref_image = None
        self.hash_threshold = hash_threshold
        self.timing = timing
        self.save_all = save_all
        self.blur_on = blur_on
        self.debug_mode = debug_mode
        self._speed = []
        self.images = []
        self.locations = []
        self._lock = Lock()
        self._skipped_frames = 0
        self.resolution = resolution
        self.bus_service_mock = bus_service_mock
        self._now = time.time()

        if detect_all:
            self.filter_classes = model.classes
        elif detect_bin_and_overflow_bin:
            self.filter_classes = ['Bin', 'Overflowed']
        else:
            self.filter_classes = DETECTOR_CLASSES_TARGETS

        if filter_duplicates:
            self.brisk_filter = BriskFilter()

    def run_inference(self, data):

        successfully_acquired = self._lock.acquire(False)
        if not successfully_acquired:
            self._skipped_frames += 1
            return
        if len(self.images) == 0:
            # first image
            self._now = time.time()
        if type(data) == str:
            data = extract_data_from_json(data, self.resolution)
        self.images.append(data['image'])
        self.locations.append({k: data[k] for k in data if k != 'image'})
        if len(self.images) < PARALLEL_IMAGES:
            self._lock.release()
            return

        if self.wait_policy.wait_interval > 0:
            logger.info('apply wait policy of %d seconds' % self.wait_policy.wait_interval)
            self.wait_policy.wait()
        now = time.time()
        logger.info('acquired data from bus')
        results = self.model_service.apply_model(self.images, self.locations)
        self._process_detections(results)
        seconds_inference = (time.time() - now)
        seconds_total = (time.time() - self._now)
        self._update_times(seconds_total, seconds_inference, PARALLEL_IMAGES, self.wait_policy.wait_interval)
        self._skipped_frames = 0
        self.images, self.locations = [], []
        self._lock.release()

    def front_image_subscribe_handler(self, data):
        Thread(target=self.run_inference, args=(data,)).start()

    def run(self):
        if self.bus_service_mock is None:
            subscribe_camera_feed('bin_overflow_front', data_handler=self.front_image_subscribe_handler,
                                  camera=Channels.FRONT_CAMERA)
            while True:
                time.sleep(300)
        else:
            while True:
                self.run_inference(self.bus_service_mock.fetch_data())

    def is_duplicate(self, frame, boxes=None, filter_brisk=False):
        is_duplicate = False
        if filter_brisk:
            brisk_thr = 2 * len(boxes)
            current_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            if self.ref_image is not None:
                matches = self.brisk_filter.get_number_of_matches(self.ref_image, self.ref_boxes, current_frame, boxes)
                if matches > brisk_thr:
                    is_duplicate = True

            self.ref_boxes = boxes
            self.ref_image = current_frame
        else:
            current_frame_hash = dhash(frame)
            if self.ref_hash is not None:
                img_dist = hash_dist(self.ref_hash, current_frame_hash)
                if img_dist < self.hash_threshold:
                    is_duplicate = True
            self.ref_hash = current_frame_hash

        return is_duplicate

    def _process_detections(self, results):
        for image, detections, location in zip(self.images, results, self.locations):
            filtered_result = [r for r in detections if r['class'] in self.filter_classes]
            res_image = write_bb(image, filtered_result, write_class=True, blur_on=self.blur_on,
                                 write_distance=True,
                                 write_bin_type=True)
            timestamp = '_'.join((time.strftime("%Y%m%d-%H%M%S"), str(datetime.datetime.now().microsecond)))

            if self.debug_mode:
                img_filename = "{}/{}.jpeg".format(DEBUG_PATH_FRAMES, timestamp)
                cv2.imwrite(img_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), self.img_quality])

            if self.save_all:
                self._save_detection(res_image, filtered_result, location['lat'], location['long'], timestamp,
                                     location['speed'], location['track'],
                                     location['is_valid_gps'])
            elif filtered_result:
                logger.info('hit on model')
                if self.filter_duplicates:
                    boxes = np.array(list(map(lambda f: f['box'], filtered_result)))
                    duplicate_frame = self.is_duplicate(res_image, boxes=boxes,
                                                        filter_brisk=True) or self.is_duplicate(
                        res_image)
                    if duplicate_frame:
                        logger.info("Duplicate hit!")
                        continue
                logger.info('saving json results')
                # When we hit detections, we save both the frame(jpg) and the results dictionary (json)
                # in their corresponding folders

                # Save separate crop and metadata for each detected object
                self._save_detection(res_image, filtered_result, location['lat'], location['long'], timestamp,
                                     location['speed'], location['track'],
                                     location['is_valid_gps'])

    def _update_times(self, execution_time_total, execution_time_inference, parallel_images, wait_interval):
        if len(self._speed) == 20:
            self._speed = self._speed[1:]
        self._speed.append(execution_time_inference)

        average_speed = sum(self._speed) / len(self._speed)

        if self.timing:
            logger.info(
                "Total processing time for this batch of %d images is %.2f seconds (wait period of %.1f seconds)" % (
                    parallel_images, execution_time_total, wait_interval))
            logger.info("Average inference time is %.2f seconds (%.2f FPS); Skipped %d frames" % (
                average_speed, parallel_images / average_speed, self._skipped_frames))

    def _write_json(self, json_filename, detection_metadata, latitude, longitude, speed, track, valid_gps_flag,
                    image_b64):
        tdm = TrashDetectionModel()
        tdm.Latitude = str(latitude)
        tdm.Longitude = str(longitude)
        tdm.Speed = str(speed)
        tdm.Bearing = str(track)
        tdm.IsValidGps = valid_gps_flag
        tdm.ModelVersion = self.model_service.model.model_version
        tdm.Image = image_b64

        if self.rescale_factor != 1.0:
            detection_metadata['box'] = tuple(int(c * self.rescale_factor) for c in detection_metadata['box'])

        tdm.ModelArtifacts = json.dumps(detection_metadata, default=jdefault)
        with open(json_filename, "w") as fout_json:
            json_data = json.dumps(tdm, default=jdefault)
            fout_json.write(json_data)

    def _save_detection(self, res_image, detections, latitude, longitude, timestamp, speed, track, valid_gps_flag):

        img_filename = "{}/{}.jpeg".format(FRAMES_PATH, timestamp)
        json_filename = "{}/{}.json".format(UPLOAD_IMAGE_PATH, timestamp)
        data_filename = "{}/{}.json".format(RESULTS_PATH, timestamp)
        image_b64 = encode_b64(res_image)

        cv2.imwrite(img_filename, cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.img_quality])

        self._write_json(json_filename, detections, latitude, longitude, speed, track, valid_gps_flag, image_b64)

        with open(data_filename, 'w') as fout:
            json.dump(detections, fout)

        if self.debug_mode:
            json_filename = "{}/{}.json".format(DEBUG_PATH_DETECTIONS, timestamp)
            with open(json_filename, 'w') as fout:
                json.dump(detections, fout)

            img_filename = "{}/{}.jpeg".format(DEBUG_PATH_RESULTS, timestamp)
            cv2.imwrite(img_filename, cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.img_quality])
