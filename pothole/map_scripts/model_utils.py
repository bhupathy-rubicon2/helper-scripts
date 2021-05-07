import math
import hashlib

import ast
import base64
import cv2
import json
import logging
import numpy as np
import os
import skimage.io
import skimage.transform
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from skimage import img_as_ubyte

from settings import RESOURCES_DIR

logger = logging.getLogger(__name__)
SMALLFONT = ImageFont.truetype(os.path.join(RESOURCES_DIR, 'fonts', 'arial.ttf'), 18)

DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 180 / math.pi
EARTH_RADIUS = 6378137.0


def bifocal_view(img):
    height, width, _ = img.shape
    img_left = img[:, :height]
    img_right = img[:, width - height:]

    return img_left, img_right


def calculate_dpos(latitude, longitude, dist, head):
    """
    Function to calculate GPS coordinates of object located at distance 'dist' and heading angle 'head' from
    truck (latitude, longitude) reference position
    
    :param latitude: GPS latitude coordinate of reference position (truck/camera)
    :param longitude: GPS longitude coordinate of reference position (truck/camera)
    :param dist: distance to the object in meters
    :param head: an angle (in degrees), to the right or left, determined by the direction of the object
    :return: (latitude, longitude) coordinates of object
    """
    latA = latitude * DEGREES_TO_RADIANS
    lonA = longitude * DEGREES_TO_RADIANS
    dist = dist / EARTH_RADIUS
    head = head * DEGREES_TO_RADIANS

    lat = math.asin(math.sin(latA) * math.cos(dist) + math.cos(latA) * math.sin(dist) * math.cos(head))

    dlon = math.atan2(math.sin(head) * math.sin(dist) * math.cos(latA),
                      math.cos(dist) - math.sin(latA) * math.sin(lat))
    lon = ((lonA + dlon + math.pi) % (math.pi * 2)) - math.pi

    return lat * RADIANS_TO_DEGREES, lon * RADIANS_TO_DEGREES


def dhash(img, size=8):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (size + 1, size + 1))

    img_h = img[:-1, :]
    hash_h = img_h[:, 1:] > img_h[:, :-1]

    img_t = img[:, :-1].T
    hash_v = img_t[:, 1:] > img_t[:, :-1]
    return np.concatenate((hash_h.ravel(), hash_v.ravel())).astype(dtype=np.uint8)


def hash_dist(hash1, hash2):
    return np.sum(hash1 != hash2)


def list_difference(list1, list2):
    return [e for e in list1 if e not in list2]


def parse_classes_file(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def parse_info_file(path):
    variables = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                name, var = line.partition(":")[::2]
                variables[name.strip()] = ast.literal_eval(var.strip())
            except Exception:
                raise ValueError('Invalid line %s in info file %s' % (line.strip(), path))
    return variables


def split_mov_into_frames(mov_path, save_to, fps=1):
    os.system('ffmpeg -i \'' + mov_path + '\' -qmin 1 -qscale:v 1 -r ' + str(fps) + " " + save_to + '/imagex-%07d.jpg')


def non_max_suppression(boxes, overlap_threshold):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    picked_indices = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        picked_indices.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

    return picked_indices


def write_bb(img, detections, write_class, blur_on, write_distance, write_bin_type):
    def draw_rectangle(draw, coordinates, color, width=1):
        for i in range(width):
            rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
            rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
            draw.rectangle((rect_start, rect_end), outline=color)

    if blur_on:
        res = Image.fromarray(img_as_ubyte(cv2.blur(img, (27, 27))))
    else:
        res = Image.fromarray(img)

    draw = ImageDraw.Draw(res)
    for d in detections:
        draw_rectangle(draw, ((d['box'][0], d['box'][1]), (d['box'][2], d['box'][3])), color=(0, 0, 255, 255),
                       width=5)

        if blur_on:
            crop_image = Image.fromarray(img[d['box'][1]: d['box'][3], d['box'][0]:d['box'][2]])
            res.paste(crop_image, (d['box'][0], d['box'][1]))

        class_name = d['class']
        if write_bin_type and 'type' in d:
            class_name += " " + d['type']

        if write_class:

            draw.text((int(d['box'][0]), int(d['box'][3] - 20)),
                      '{:s} {:.1f}'.format(class_name, d['score']), font=SMALLFONT, fill=(0, 0, 255, 255))

            if 'distance' in d and 'angle' in d and write_distance:
                draw.text((int(d['box'][0]), int(d['box'][3] - 35)),
                          '{:s} {:.2f}m {:s} {:.2f}degrees'.format("Dist", d['distance'], "Angle", d['angle']),
                          font=SMALLFONT, fill=(0, 0, 255, 255))

    return np.array(res)


def encode_b64(img):
    img = Image.fromarray(img)
    buffered = BytesIO()
    img.save(buffered, format='jpeg')
    img_str = str(base64.b64encode(buffered.getvalue()))
    return img_str


def decode_b64(img):
    img = base64.b64decode(img)
    img = Image.open(BytesIO(img))
    img = np.array(img)
    return img


def resize_image(img, resolution):
    if resolution == '1080p':
        if img.shape != (1080, 1920, 3):
            img = skimage.transform.resize(img, (1080, 1920, 3), preserve_range=True).astype(np.uint8)
    elif resolution == '720p':
        if img.shape != (720, 960, 3):
            img = skimage.transform.resize(img, (720, 960, 3), preserve_range=True).astype(np.uint8)
    else:
        raise ValueError("Invalid camera resolution: " + resolution)
    return img


def extract_data_from_json(data, resolution):
    data = json.loads(data)
    img = data.get('base64_image', None)
    if img is None:
        raise ValueError("Couldn't fetch data from service bus")
    img = decode_b64(img)
    img = resize_image(img, resolution)
    location = data.get('location', {})
    return {
        'lat': location.get('Latitude', 0),
        'long': location.get('Longitude', 0),
        'is_valid_gps': location.get('IsValidGps', 0),
        'speed': location.get('Speed', 0),
        'track': location.get('Bearing', 0),
        'image': img,
        'frame_rate': data.get('frame_rate', 0)
    }

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