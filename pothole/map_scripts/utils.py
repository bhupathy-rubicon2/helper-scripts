import numpy as np
import os

from skimage.color import rgba2rgb
from skimage.transform import resize, rescale

LOOKUP = {'overflowed': 'bin_pos',
          'bin other': 'bin_other',
          'bin': 'bin_neg',
          'ahead only': 'traffic_sign_ahead_only',
          'caution children': 'traffic_sign_caution_children',
          'crosswalk': 'traffic_sign_crosswalk',
          'school crosswalk': 'traffic_sign_school_crosswalk',
          'dead end': 'traffic_sign_dead_end',
          'no parking': 'traffic_sign_no_parking',
          'speed limit 25': 'traffic_sign_speed_limit_25',
          'speed limit 30': 'traffic_sign_speed_limit_30',
          'speed limit 35': 'traffic_sign_speed_limit_35',
          'stop': 'traffic_sign_stop',
          'stop ahead': 'traffic_sign_stop_ahead',
          'lights': 'traffic_sign_traffic_lights',
          'yield': 'traffic_sign_yield',
          'trash object': 'trash_object',
          'waste container': 'waste_container'
          }


def class_lookup(cls_name):
    inv_map = {v: k for k, v in LOOKUP.items()}
    return inv_map.get(cls_name, cls_name)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def crop_box(image, box, new_size):
    if image.shape[-1] == 4:
        image = rgba2rgb(image)
        image = (image * 255).astype(np.uint8)

    height, width, _ = image.shape
    xmin, xmax, ymin, ymax = box[0], box[1], box[2], box[3]
    x_size = xmax - xmin
    y_size = ymax - ymin
    if x_size < new_size:
        xmin -= int((new_size - x_size) / 2)
        xmax += int((new_size - x_size) / 2)
    else:
        # extend box
        xmin = int(xmin - 0.01 * x_size)
        xmax = int(xmax + 0.01 * x_size)
    if y_size < new_size:
        ymin -= int((new_size - y_size) / 2)
        ymax += int((new_size - y_size) / 2)
    else:
        ymin = int(ymin - 0.01 * y_size)
        ymax = int(ymax + 0.01 * y_size)

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)
    new_image = image.copy()[ymin: ymax, xmin: xmax]
    new_image = resize(new_image, (new_size, new_size), preserve_range=True)

    return new_image.astype(np.uint8)


def compute_class_weights(path_to_train_data):
    weights = {}
    for cls in os.listdir(path_to_train_data):
        if os.path.isdir(os.path.join(path_to_train_data, cls)):
            weights[cls] = len(os.listdir(os.path.join(path_to_train_data, cls)))
    ref = max(weights.values())
    for k, v in weights.items():
        weights[k] = ref / v
    return weights


def bifocal_view(img, coords=None):
    height, width, _ = img.shape
    img_left = img[:, :height]
    img_right = img[:, width - height:]
    coords_left = []
    coords_right = []

    if coords is not None:

        for coord in coords:
            class_, xmin, ymin, xmax, ymax = coord
            if xmin < img_left.shape[1]:
                coords_left.append([class_, xmin, ymin, min(xmax, height), ymax])
            if xmax > width - height:
                xmin = max(xmin - (width - height), 1)
                xmax = xmax - (width - height)
                coords_right.append([class_, xmin, ymin, xmax, ymax])

    return img_left, img_right, coords_left, coords_right


def rescale_image(img, dimension):
    if img.shape[-1] == 4:
        img = rgba2rgb(img)
        img = (img * 255).astype(np.uint8)

    max_dim = max(img.shape[0], img.shape[1])
    scaled_img = rescale(img, float(dimension) / max_dim, preserve_range=True)

    shp = scaled_img.shape
    left_pad = int(round(float((dimension - shp[0])) / 2))
    right_pad = int(round(float(dimension - shp[0]) - left_pad))
    top_pad = int(round(float((dimension - shp[1])) / 2))
    bottom_pad = int(round(float(dimension - shp[1]) - top_pad))
    pads = ((left_pad, right_pad), (top_pad, bottom_pad))

    new_image = np.zeros((dimension, dimension, img.shape[-1]), dtype=np.float32)

    for i in range(new_image.shape[-1]):
        new_image[:, :, i] = np.lib.pad(scaled_img[:, :, i], pads, 'constant', constant_values=((0, 0), (0, 0)))

    return new_image


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
