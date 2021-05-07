import cv2
import numpy as np


def determine_bin_type(img_raw, bounding_box=None, bin_color_dict=None, rgb_format=False):
    """

    :param img_raw: an image ingested via opencv and stored in BGR format. Optionally as rgb 
    if rgb_format is true

    :param bounding_box: optional argument specifying a bounding box in the image
    : format is [x,y,w,h]

    :param bin_color_dict: optional argument specifying a dictionary of HSV color values
    : key is bin type
    : value is HSV color

    :param rgb_format: image is stored in rgb format

    """

    if bounding_box is not None:
        x = bounding_box[0]
        y = bounding_box[1]
        w = bounding_box[2]
        h = bounding_box[3]

        img = img_raw[y:y + h, x:x + w]

    else:
        img = img_raw

    # Bin colors in HSV space
    if bin_color_dict is None:
        bin_color_dict = {}
        bin_color_dict['recycle'] = [116, 255, 255]  # Blue bin, blue top
        bin_color_dict['organics'] = [75, 255, 255]  # Green bin, green top
        bin_color_dict['trash'] = [0, 0, 0]  # Black bin, black top

    # Convert image to HSV color space
    if rgb_format:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # This assumes that the background is mostly white
    # So that the background has HSV=(0,0,255)
    hue_val = np.percentile(hsv[:, :, 0], 80)
    sat_val = np.percentile(hsv[:, :, 1], 80)
    val_val = np.percentile(hsv[:, :, 2], 20)

    min_distance = 1e8
    bin_type = None
    for key in bin_color_dict.keys():

        distance = np.sqrt(
            min(
                (hue_val - bin_color_dict[key][0]) ** 2,
                (hue_val + 360 - bin_color_dict[key][0]) ** 2,
                (hue_val - 360 - bin_color_dict[key][0]) ** 2
            ) + \
            (sat_val - bin_color_dict[key][1]) ** 2 + \
            (val_val - bin_color_dict[key][2]) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            bin_type = key

    return bin_type
