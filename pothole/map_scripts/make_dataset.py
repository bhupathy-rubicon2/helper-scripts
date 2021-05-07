from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import glob
import io
import numpy as np
import pandas as pd
import random
import shutil
import skimage.io
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
from google.protobuf import text_format
from skimage.transform import rescale, resize

from config import *
from utils import get_iou, crop_box, bifocal_view
import string_int_label_map_pb2

HEAVY_AUG = False

MAKE_CROPS = False
MAKE_PREFILTER = False

HEAVY_AUG_CROPS = False
HEAVY_AUG_PREFILTER = False

#CLASSES = None

PREFILTER_CLASSES = ['positive', 'negative']
CROP_SIZE = 224

PREFILTER_SIZE = 224

SCALES = [0.75, 1, 1.25, 1.5]

SPLIT_PERCENTAGE_CROPS = 0.75

RATIO_THRESHOLD = 0.1


def create_test_gt_files(tst_dir, write_path):
    """
    Create the ground truth files needed for evaluation
    :param tst_dir: test directory
    :param write_path: ground truth files directory
    :return: None
    """
    os.makedirs(write_path, exist_ok=True)
    for file in glob.glob(os.path.join(tst_dir, 'annotations', '*.xml')):
        tree = ET.parse(file)
        root = tree.getroot()

        rows = []
        fname = None
        for member in root.findall('object'):
            fname = root.find('filename').text
            cls = member[0].text
            difficult = " difficult" if member[3].text == "1" else ""

            if AGGREGATE_CLASSES and cls in CLASSES_CORRESP:
                cls = CLASSES_CORRESP[cls]

            if cls not in CLASSES:
                continue

            rows.append(cls + " " + member[4][0].text + " " +
                        member[4][1].text + " " + member[4][2].text + " " +
                        member[4][3].text + difficult
                        )

        if len(rows) > 0:
            with open(os.path.join(write_path, fname.replace('.jpg', '.txt')), 'w') as f:
                for r in rows:
                    f.writelines(r + '\n')


def xml_to_csv(path):
    """
    Save to csv the instances
    :param path: path of dataset
    :return: None
    """
    count = dict(zip(CLASSES, [0] * len(CLASSES)))
    print(CLASSES)
    xml_list = []
    img_path = os.path.join(path, 'img')
    xml_path = os.path.join(path, 'annotations')

    xmls = os.listdir(xml_path)
    random.shuffle(xmls)

    for i, file in enumerate(xmls):
        xml_file = os.path.join(xml_path, file)
        img_file = os.path.join(img_path, file.replace(".xml", ".jpg"))
        if os.path.exists(img_file):
            print(xml_file)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                cls = member[0].text

                if cls.startswith('waste_container'):
                    cls = 'waste_container'

                if AGGREGATE_CLASSES and cls in CLASSES_CORRESP:
                    cls = CLASSES_CORRESP[cls]

                if cls not in CLASSES:
                    continue

                value = (img_path,
                         root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         cls,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
                count[cls] += 1

    column_name = ['path', 'filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print("Instances in %s:\n%s" % (path, count))
    return xml_df


def load_labelmap(path):
    """
    Load labelmap
    :param path: path to pb file
    :return: label map dictionary and list of keys
    """
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()

        label_map = string_int_label_map_pb2.StringIntLabelMap()
        text_format.Merge(label_map_string, label_map)

    label_map = {item.name: item.id for item in label_map.item}
    return label_map, list(label_map.keys())


def split(df, group):
    """
    Group dataframe
    :param df: data frame
    :param group: column name on which to group
    :return: list of groups
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def class_text_to_int(cls_name, label_map):
    """
    Get index of class name
    :param cls_name: name of class
    :param label_map: label map
    :return: index of class if found
    """
    if cls_name in label_map:
        return label_map[cls_name]
    raise ValueError('Invalid class')


def create_tf_example(group, path, label_map):
    """
    Create tf example for images based on groups of records
    :param group: group for records
    :param path: path for image
    :param label_map: map for classes
    :return: sample as tf Example
    """
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_list_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def bytes_list_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map=label_map))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def write_tfrecords(input_csv, output, label_map):
    """
    Write from csv to tfrecords
    :param input_csv: csv file
    :param output: output file for tfrecords
    :param label_map: map for classes
    :return: None
    """
    writer = tf.python_io.TFRecordWriter(output)
    examples = pd.read_csv(input_csv)
    grouped = split(examples, 'filename')
    random.shuffle(grouped)
    for group in grouped:
        tf_example = create_tf_example(group, group.object.path.values[0], label_map=label_map)
        writer.write(tf_example.SerializeToString())
    writer.close()


def create_crops(input_csv, output, classes, size):
    """
    Crop boxes for objects in an image
    :param input_csv: csv file with samples
    :param output: output directory
    :param classes: classes list
    :param size: size of output
    :return: None
    """
    for cls in classes:
        if not os.path.exists(os.path.join(output, cls)):
            os.makedirs(os.path.join(output, cls))

    grouped = split(pd.read_csv(input_csv), 'filename')
    for group in grouped:
        boxes = []
        image = skimage.io.imread(os.path.join(DETECTION_DATA_DIR, 'img', group.filename), plugin='imageio')
        height, width, _ = image.shape

        for index, row in group.object.iterrows():
            cls = row['class']
            cropped_image = crop_box(image, (row['xmin'], row['xmax'], row['ymin'], row['ymax']), new_size=size)

            skimage.io.imsave(os.path.join(output, cls, group.filename.replace('.jpg', '_' + str(index) + '.jpg')),
                              cropped_image, plugin='imageio')

            boxes.append((row['xmin'], row['xmax'], row['ymin'], row['ymax']))

        extracted_negatives = 0
        failures = 0
        to_extract = np.random.choice(2)

        while extracted_negatives < to_extract and failures < 10:
            scale = SCALES[np.random.randint(0, len(SCALES))]
            img = rescale(image, scale, preserve_range=True).astype(np.uint8)
            if img.shape[0] > size and img.shape[1] > size:
                y = np.random.randint(0, img.shape[0] - size)
                x = np.random.randint(0, img.shape[1] - size)
            else:
                continue

            bbox = {'x1': int(x), 'x2': int(x + size), 'y1': int(y), 'y2': int(y + size)}

            overlap = False
            for box in boxes:
                x1, x2, y1, y2 = box

                bb1 = {'x1': int(x1 * scale), 'x2': int(x2 * scale),
                       'y1': int(y1 * scale), 'y2': int(y2 * scale)}

                if get_iou(bbox, bb1) != 0:
                    overlap = True
                    break

            if not overlap:
                cropped_image = img[y:y + size, x:x + size]
                skimage.io.imsave(
                    os.path.join(output, classes[0],
                                 group.filename.replace('.jpg', '_' + str(extracted_negatives) + '.jpg')),
                    cropped_image, plugin='imageio')
                extracted_negatives += 1
            else:
                failures += 1


def move_crops(input_dir, train_dir, val_dir, size, classes, per_split=0.75):
    """
    Move crops and split them to train set and validation set
    :param input_dir: path towards input directory
    :param train_dir: path towards train directory
    :param val_dir: path towards validation directory
    :param size: size for image
    :param classes: list of classes
    :param per_split: percentage kept in train set
    :return: None
    """
    for cls in classes:
        if not os.path.exists(os.path.join(train_dir, cls)):
            os.makedirs(os.path.join(train_dir, cls))
        if not os.path.exists(os.path.join(val_dir, cls)):
            os.makedirs(os.path.join(val_dir, cls))
    for cls in classes:
        images = glob.glob(os.path.join(input_dir, cls, '*.jpg'))
        stop = int(per_split * len(images))
        random.shuffle(images)
        train_data = images[:stop]
        val_data = images[stop:]

        for fname in train_data:
            image = skimage.io.imread(fname)
            image = resize(image, (size, size))
            skimage.io.imsave(os.path.join(train_dir, cls, os.path.basename(fname)), image)

        for fname in val_data:
            image = skimage.io.imread(fname)
            image = resize(image, (size, size))
            skimage.io.imsave(os.path.join(val_dir, cls, os.path.basename(fname)), image)


def fullhd_to_bifocal(df, tmp_folder="/tmp/bifocal_crops"):
    """
    Create bifocal crops
    :param df: dataframe with initial data
    :param tmp_folder: temporary folder where the crops should be saved
    :return: dataframe with bifocal samples
    """
    grouped = split(df, 'filename')
    res_list = []

    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    for group in grouped:
        img_left, img_right, coords_left, coords_right = bifocal_view(
            skimage.io.imread(os.path.join(group.object.path.values[0], group.filename)),
            group.object[['class', 'xmin', 'ymin', 'xmax', 'ymax']].values
        )
        img_left_name = "1_{}".format(group.filename)
        img_right_name = "2_{}".format(group.filename)

        skimage.io.imsave(os.path.join(tmp_folder, img_left_name), img_left)
        skimage.io.imsave(os.path.join(tmp_folder, img_right_name), img_right)

        for idx, box in enumerate(coords_left):
            values = [tmp_folder, img_left_name, img_left.shape[0], img_left.shape[0], *box]
            res_list.append(values)

        for idx, box in enumerate(coords_right):
            values = [tmp_folder, img_right_name, img_right.shape[0], img_right.shape[0], *box]
            res_list.append(values)

    return pd.DataFrame(res_list, columns=df.columns)


def create_prefilter(input_csv, output, size, threshold_ratio=0.1):
    """
    Create prefilter data
    :param input_csv: input csv
    :param output: output directory
    :param size: size for images
    :param threshold_ratio: threshold for shape ratio
    :return: None
    """
    grouped = split(pd.read_csv(input_csv), 'filename')

    for cls in PREFILTER_CLASSES:
        if not os.path.exists(os.path.join(output, cls)):
            os.makedirs(os.path.join(output, cls))

    # read each image once
    for group in grouped:
        image = skimage.io.imread(os.path.join(DETECTION_DATA_DIR, 'img', group.filename), plugin='imageio')
        height, width, _ = image.shape

        ratio = width / height
        if abs(ratio - 1) < threshold_ratio:
            # resize the image
            new_image = resize(image, (size, size), preserve_range=True).astype(np.uint8)
            skimage.io.imsave(os.path.join(output, PREFILTER_CLASSES[0], group.filename), new_image, plugin='imageio')


def main(_):
    global LABEL_MAP, CLASSES
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DETECTION_DIR)

    training_labels_csv = os.path.join(PROCESSED_DETECTION_DIR, 'training-labels.csv')
    training_records = os.path.join(PROCESSED_DETECTION_DIR, 'train.record')
    val_labels_csv = os.path.join(PROCESSED_DETECTION_DIR, 'val-labels.csv')
    val_records = os.path.join(PROCESSED_DETECTION_DIR, 'val.record')
    test_labels_csv = os.path.join(PROCESSED_DETECTION_DIR, 'test-labels.csv')

    LABEL_MAP, CLASSES = load_labelmap(DETECTION_CLASSES_FILE)

    #xml_df_training = xml_to_csv(DETECTION_DATA_DIR)
    #xml_df_training.to_csv(training_labels_csv, index=None)

    #xml_df_val = xml_to_csv(DETECTION_VAL_DIR)
    #xml_df_val.to_csv(val_labels_csv, index=None)

    xml_df_test = xml_to_csv(TEST_DIR)
    create_test_gt_files(TEST_DIR, write_path=TEST_GROUND_TRUTH_PATH)

    xml_df_test.to_csv(test_labels_csv, index=None)

    print('Successfully converted xml to csv.')

    #write_tfrecords(val_labels_csv, val_records, label_map=LABEL_MAP)

    # augment the training dataset
    if HEAVY_AUG:
        from data.augmentation import augment_detection_data
        import multiprocessing

        nproc = os.cpu_count()

        manager = multiprocessing.Manager()

        dfs = [manager.Value(pd.DataFrame, pd.DataFrame()) for _ in range(nproc)]
        xml_df_training_aug = pd.DataFrame()
        procs = []

        n = len(xml_df_training.index)

        if os.path.exists(AUG_DIR):
            shutil.rmtree(AUG_DIR)
        os.makedirs(AUG_DIR)

        for i in range(nproc):
            print(int(i * n / nproc), int((i + 1) * n / nproc))
            p = multiprocessing.Process(target=augment_detection_data, args=(
                xml_df_training[int(i * n / nproc):int((i + 1) * n / nproc)], AUG_DIR, dfs[i]))
            procs.append(p)
            p.start()
            print('Started process ', i)

        for idx, p in enumerate(procs):
            p.join()
            print('Joined process ', idx)
            xml_df_training_aug = pd.concat([xml_df_training_aug, dfs[idx].value], ignore_index=True)

        training_aug_labels_csv = os.path.join(PROCESSED_DETECTION_DIR, 'training_aug-labels.csv')
        print("Total augmented samples %d" % len(xml_df_training_aug.index))
        xml_df_training_aug.to_csv(training_aug_labels_csv, index=None)
        write_tfrecords([training_labels_csv, training_aug_labels_csv], training_records, label_map=LABEL_MAP)
    #else:
    #    write_tfrecords(training_labels_csv, training_records, label_map=LABEL_MAP)

    print('Successfully created tf records.')

    if MAKE_CROPS:
        if os.path.exists(PROCESSED_CROPS_DIR):
            shutil.rmtree(PROCESSED_CROPS_DIR)

        filter_train_dir = os.path.join(PROCESSED_CROPS_DIR, 'training')
        filter_val_dir = os.path.join(PROCESSED_CROPS_DIR, 'validation')
        create_crops(training_labels_csv, filter_train_dir, classes=CLASSES, size=CROP_SIZE)
        create_crops(val_labels_csv, filter_val_dir, classes=CLASSES, size=CROP_SIZE)
        move_crops(FILTER_DATA_DIR, filter_train_dir, filter_val_dir, size=CROP_SIZE,
                   classes=['negative'] + CLASSES, per_split=SPLIT_PERCENTAGE_CROPS)

        if HEAVY_AUG_CROPS:
            from data.augmentation import augment_filter_data
            augment_filter_data(filter_train_dir, aug_degree=3)
        print('Successfully created crops.')

    if MAKE_PREFILTER:
        if os.path.exists(PROCESSED_PREFILTER_DIR):
            shutil.rmtree(PROCESSED_PREFILTER_DIR)

        prefilter_training = os.path.join(PROCESSED_PREFILTER_DIR, 'training')
        prefilter_validation = os.path.join(PROCESSED_PREFILTER_DIR, 'validation')
        create_prefilter(training_labels_csv, prefilter_training,
                         size=PREFILTER_SIZE, threshold_ratio=RATIO_THRESHOLD)
        create_prefilter(val_labels_csv, prefilter_validation,
                         size=PREFILTER_SIZE, threshold_ratio=RATIO_THRESHOLD)
        move_crops(PREFITLER_DATA_DIR, prefilter_training, prefilter_validation, size=PREFILTER_SIZE,
                   classes=PREFILTER_CLASSES, per_split=SPLIT_PERCENTAGE_CROPS)
        if HEAVY_AUG_PREFILTER:
            from data.augmentation import augment_filter_data
            augment_filter_data(prefilter_training, base_prob=0.6, aug_degree=3)
        print('Successfully created crops.')
        print('Successfully created prefilter dataset.')


if __name__ == '__main__':
    tf.app.run()
