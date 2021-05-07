from time import time

import argparse
import glob
import numpy as np
import os
import shutil
import skimage.io

from config import TEST_DETECTIONS_PATH, MODELS_DIR, TEST_IMAGES_PATH, TEST_GROUND_TRUTH_PATH
from utils import LOOKUP
from model_utils import parse_classes_file, parse_info_file, list_difference, get_model_hash

MODELS_PATH = os.path.join(MODELS_DIR, '')
PREDICTIONS_PATH = '/tmp/predictions'


class SSD:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'ssd-tf', 'detector.pb')
    INFO_PATH = os.path.join(MODELS_PATH, 'ssd-tf', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'ssd-tf', 'z-classes.names')


class SSD_FPN:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-tf', 'detector.pb')
    INFO_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-tf', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'ssd-fpn-tf', 'z-classes.names')


class SSD_FPN_OPENVINO:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-openvino', 'detector.bin')
    CFG_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-openvino', 'detector.xml')
    INFO_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-openvino', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'ssd-fpn-openvino', 'z-classes.names')


class SSD_FPN_OPENVINO_CPU:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-openvino-cpu', 'detector.bin')
    CFG_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-openvino-cpu', 'detector.xml')
    INFO_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-openvino-cpu', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'ssd-fpn-openvino-cpu', 'z-classes.names')


class SSD_FPN_OPENVINO_CPU:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-openvino-cpu', 'detector.bin')
    CFG_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-openvino-cpu', 'detector.xml')
    INFO_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-openvino-cpu', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'ssd-fpn-openvino-cpu', 'z-classes.names')


class SSD_FPN_RESNET:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-resnet-tf', 'detector.pb')
    INFO_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-resnet-tf', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'ssd-fpn-tf', 'z-classes.names')


class SSD_FPN_RESNET_OPENVINO:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-resnet-openvino', 'detector.bin')
    CFG_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-resnet-openvino', 'detector.xml')
    INFO_PATH = os.path.join(MODELS_PATH, 'ssd-fpn-resnet-openvino', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'ssd-fpn-resnet-openvino', 'z-classes.names')


class FRCNN_INCEPTION_OPENVINO:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'frcnn-inception-openvino', 'detector.bin')
    CFG_PATH = os.path.join(MODELS_PATH, 'frcnn-inception-openvino', 'detector.xml')
    INFO_PATH = os.path.join(MODELS_PATH, 'frcnn-inception-openvino', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'frcnn-inception-openvino', 'z-classes.names')


class FRCNN_INCEPTION_TF:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'frcnn-inception-tf', 'detector.pb')
    INFO_PATH = os.path.join(MODELS_PATH, 'frcnn-inception-tf', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'frcnn-inception-tf', 'z-classes.names')


class FRCNN_RESNET_TF:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'frcnn-resnet-tf', 'detector.pb')
    INFO_PATH = os.path.join(MODELS_PATH, 'frcnn-resnet-tf', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'frcnn-resnet-tf', 'z-classes.names')


class MXNET:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'yolo-mxnet', 'detector.params')
    INFO_PATH = os.path.join(MODELS_PATH, 'yolo-mxnet', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'yolo-mxnet', 'z-classes.names')


class YOLO:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'yolo-cv2', 'detector.weights')
    CFG_PATH = os.path.join(MODELS_PATH, 'yolo-cv2', 'detector.cfg')
    INFO_PATH = os.path.join(MODELS_PATH, 'yolo-cv2', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'yolo-cv2', 'z-classes.names')


class SSD_INCEPTION_TF:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'ssd-inception-tf', 'detector.pb')
    INFO_PATH = os.path.join(MODELS_PATH, 'ssd-inception-tf', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'ssd-inception-tf', 'z-classes.names')


class SSD_INCEPTION_OPENVINO:
    DETECTOR_PATH = os.path.join(MODELS_PATH, 'ssd-inception-openvino', 'detector.bin')
    CFG_PATH = os.path.join(MODELS_PATH, 'ssd-inception-openvino', 'detector.xml')
    INFO_PATH = os.path.join(MODELS_PATH, 'ssd-inception-openvino', 'detector.info')
    CLASSES_NAMES = os.path.join(MODELS_PATH, 'ssd-inception-openvino', 'z-classes.names')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a bin detection network')
    parser.add_argument('--model', dest='model', help='model type (ssd, yolo, mxnet)', type=str, default='ssd')
    parser.add_argument('--bifocal', dest='bifocal', help='bifocal (0 or 1)', type=int, default=0)
    parser.add_argument('--bin_only', dest='bin_only', help='bin_only (0 or 1)', type=int, default=0)

    args = parser.parse_args()
    return args


def gt_file_has_class(file_path, cls_name):
    lines = parse_classes_file(file_path)
    for l in lines:
        if l.split(' ')[0] == cls_name:
            return True
    return False


if __name__ == '__main__':
    if os.path.exists(TEST_DETECTIONS_PATH):
        shutil.rmtree(TEST_DETECTIONS_PATH)
    os.makedirs(TEST_DETECTIONS_PATH)

    args = parse_args()
    if args.model == 'ssd':
        from model.ssd_detector import BinDetectorSSD

        info = parse_info_file(SSD.INFO_PATH)
        classes = parse_classes_file(SSD.CLASSES_NAMES)
        model = BinDetectorSSD(SSD.DETECTOR_PATH, classes=classes,
                               threshold=info['THRESHOLD'],
                               excluded_classes=list_difference(classes,
                                                                ['Overflowed', 'Bin']) if args.bin_only == 1 else [],
                               box_area_limit=info['BOX_AREA_LIMIT'],
                               model_version=info['MODEL_VERSION'])
        h = get_model_hash(SSD.DETECTOR_PATH)
        print("Loaded ssd model %s" % h)
    elif args.model == 'yolo':
        from model.yolo_detector import BinDetectorYOLO

        info = parse_info_file(YOLO.INFO_PATH)
        classes = parse_classes_file(YOLO.CLASSES_NAMES)
        model = BinDetectorYOLO(YOLO.CFG_PATH, YOLO.DETECTOR_PATH, classes=classes,
                                threshold=info['THRESHOLD'],
                                excluded_classes=list_difference(classes,
                                                                 ['Overflowed', 'Bin']) if args.bin_only == 1 else [],
                                box_area_limit=info['BOX_AREA_LIMIT'],
                                model_version=info['MODEL_VERSION'])
        h = get_model_hash(YOLO.DETECTOR_PATH)
        print("Loaded yolo model %s" % h)
    elif args.model == 'ssd_fpn':
        from model.ssd_detector import BinDetectorSSD

        classes = parse_classes_file(SSD_FPN.CLASSES_NAMES)
        info = parse_info_file(SSD_FPN.INFO_PATH)
        model = BinDetectorSSD(SSD_FPN.DETECTOR_PATH,
                               classes=classes,
                               threshold=info['THRESHOLD'],
                               excluded_classes=list_difference(classes,
                                                                ['Overflowed', 'Bin']) if args.bin_only == 1 else [],
                               box_area_limit=info['BOX_AREA_LIMIT'],
                               model_version=info['MODEL_VERSION']
                               )
        h = get_model_hash(SSD_FPN.DETECTOR_PATH)
        print("Loaded ssd fpn model %s" % h)
    elif args.model == 'mxnet':
        from model.ssd_detector import BinDetectorMxnet

        info = parse_info_file(MXNET.INFO_PATH)
        classes = parse_classes_file(MXNET.CLASSES_NAMES)
        model = BinDetectorMxnet(MXNET.DETECTOR_PATH, classes=classes,
                                 threshold=info['THRESHOLD'],
                                 excluded_classes=list_difference(classes,
                                                                  ['Overflowed', 'Bin']) if args.bin_only == 1 else [],
                                 box_area_limit=info['BOX_AREA_LIMIT'],
                                 model_version=info['MODEL_VERSION'])
        h = get_model_hash(MXNET.DETECTOR_PATH)
        print("Loaded mxnet model %s" % h)
    elif args.model == 'ssd_inception':
        from model.ssd_detector import BinDetectorSSD

        classes = parse_classes_file(SSD_INCEPTION_TF.CLASSES_NAMES)
        info = parse_info_file(SSD_INCEPTION_TF.INFO_PATH)
        model = BinDetectorSSD(SSD_INCEPTION_TF.DETECTOR_PATH,
                               classes=classes,
                               threshold=info['THRESHOLD'],
                               excluded_classes=list_difference(classes,
                                                                ['Overflowed', 'Bin']) if args.bin_only == 1 else [],
                               box_area_limit=info['BOX_AREA_LIMIT'],
                               model_version=info['MODEL_VERSION']
                               )
        h = get_model_hash(SSD_INCEPTION_TF.DETECTOR_PATH)
        print("Loaded ssd inception model %s" % h)
    elif args.model == 'ssd_inception_openvino':
        from model.ssd_openvino_detector import BinDetectorOpenVino

        classes = parse_classes_file(SSD_INCEPTION_OPENVINO.CLASSES_NAMES)
        info = parse_info_file(SSD_INCEPTION_OPENVINO.INFO_PATH)
        model = BinDetectorOpenVino(SSD_INCEPTION_OPENVINO.CFG_PATH,
                                    SSD_INCEPTION_OPENVINO.DETECTOR_PATH,
                                    classes=classes,
                                    threshold=info['THRESHOLD'],
                                    excluded_classes=list_difference(classes,
                                                                     ['Overflowed',
                                                                      'Bin']) if args.bin_only == 1 else [],
                                    box_area_limit=info['BOX_AREA_LIMIT'],
                                    model_version=info['MODEL_VERSION'],
                                    resize_h=300,
                                    resize_w=300
                                    )
        h = get_model_hash(SSD_INCEPTION_OPENVINO.DETECTOR_PATH)
        print("Loaded ssd inception open vino model %s" % h)
    elif args.model == 'ssd_fpn_openvino':
        from model.ssd_openvino_detector import BinDetectorOpenVino

        classes = parse_classes_file(SSD_FPN_OPENVINO.CLASSES_NAMES)
        info = parse_info_file(SSD_FPN_OPENVINO.INFO_PATH)
        model = BinDetectorOpenVino(SSD_FPN_OPENVINO.CFG_PATH,
                                    SSD_FPN_OPENVINO.DETECTOR_PATH,
                                    classes=classes,
                                    threshold=info['THRESHOLD'],
                                    #excluded_classes=list_difference(classes,
                                    #                                 ['Overflowed',
                                    #                                  'Bin']) if args.bin_only == 1 else [],
                                    num_requests=1,
                                    box_area_limit=info['BOX_AREA_LIMIT'],
                                    model_version=info['MODEL_VERSION'],
                                    resize_h=349,
                                    resize_w=349
                                    )
        h = get_model_hash(SSD_FPN_OPENVINO.DETECTOR_PATH)
        print("Loaded ssd fpn open vino model %s" % h)
    elif args.model == 'ssd_fpn_openvino_cpu':
        from model.ssd_openvino_detector import BinDetectorOpenVino

        classes = parse_classes_file(SSD_FPN_OPENVINO_CPU.CLASSES_NAMES)
        info = parse_info_file(SSD_FPN_OPENVINO_CPU.INFO_PATH)
        model = BinDetectorOpenVino(SSD_FPN_OPENVINO_CPU.CFG_PATH,
                                    SSD_FPN_OPENVINO_CPU.DETECTOR_PATH,
                                    classes=classes,
                                    threshold=info['THRESHOLD'],
                                    #excluded_classes=list_difference(classes,
                                    #                                 ['Overflowed',
                                    #                                  'Bin']) if args.bin_only == 1 else [],
                                    num_requests=1,
                                    box_area_limit=info['BOX_AREA_LIMIT'],
                                    model_version=info['MODEL_VERSION'],
                                    resize_h=349,
                                    resize_w=349,
                                    run_on_cpu=True
                                    )
        h = get_model_hash(SSD_FPN_OPENVINO_CPU.DETECTOR_PATH)
        print("Loaded ssd fpn open vino cpu model %s" % h)
    elif args.model == 'ssd_fpn_openvino_cpu':
        from model.ssd_openvino_detector import BinDetectorOpenVino

        classes = parse_classes_file(SSD_FPN_OPENVINO_CPU.CLASSES_NAMES)
        info = parse_info_file(SSD_FPN_OPENVINO_CPU.INFO_PATH)
        model = BinDetectorOpenVino(SSD_FPN_OPENVINO_CPU.CFG_PATH,
                                    SSD_FPN_OPENVINO_CPU.DETECTOR_PATH,
                                    classes=classes,
                                    threshold=info['THRESHOLD'],
                                    excluded_classes=list_difference(classes,
                                                                     ['Overflowed',
                                                                      'Bin']) if args.bin_only == 1 else [],
                                    box_area_limit=info['BOX_AREA_LIMIT'],
                                    model_version=info['MODEL_VERSION'],
                                    resize_h=349,
                                    resize_w=349,
                                    run_on_cpu=True
                                    )
        h = get_model_hash(SSD_FPN_OPENVINO_CPU.DETECTOR_PATH)
        print("Loaded ssd fpn open vino model for cpu %s" % h)
    elif args.model == 'ssd_fpn_resnet_openvino':
        from model.ssd_openvino_detector import BinDetectorOpenVino

        classes = parse_classes_file(SSD_FPN_RESNET_OPENVINO.CLASSES_NAMES)
        info = parse_info_file(SSD_FPN_RESNET_OPENVINO.INFO_PATH)
        model = BinDetectorOpenVino(SSD_FPN_RESNET_OPENVINO.CFG_PATH,
                                    SSD_FPN_RESNET_OPENVINO.DETECTOR_PATH,
                                    classes=classes,
                                    threshold=info['THRESHOLD'],
                                    excluded_classes=list_difference(classes,
                                                                     ['Overflowed',
                                                                      'Bin']) if args.bin_only == 1 else [],
                                    box_area_limit=info['BOX_AREA_LIMIT'],
                                    model_version=info['MODEL_VERSION'],
                                    resize_h=640,
                                    resize_w=640
                                    )
        h = get_model_hash(SSD_FPN_RESNET_OPENVINO.DETECTOR_PATH)
        print("Loaded ssd fpn resnet open vino model %s" % h)
    elif args.model == 'ssd_fpn_resnet_tf':
        from model.ssd_detector import BinDetectorSSD

        classes = parse_classes_file(SSD_FPN_RESNET.CLASSES_NAMES)
        info = parse_info_file(SSD_FPN_RESNET.INFO_PATH)
        model = BinDetectorSSD(SSD_FPN_RESNET.DETECTOR_PATH,
                               classes=classes,
                               threshold=info['THRESHOLD'],
                               excluded_classes=list_difference(classes,
                                                                ['Overflowed', 'Bin']) if args.bin_only == 1 else [],
                               box_area_limit=info['BOX_AREA_LIMIT'],
                               model_version=info['MODEL_VERSION']
                               )
        h = get_model_hash(SSD_FPN_RESNET.DETECTOR_PATH)
        print("Loaded ssd fpn resnet model %s" % h)
    elif args.model == 'frcnn_inception_openvino':
        from model.ssd_openvino_detector import BinDetectorOpenVino

        classes = parse_classes_file(FRCNN_INCEPTION_OPENVINO.CLASSES_NAMES)
        info = parse_info_file(FRCNN_INCEPTION_OPENVINO.INFO_PATH)
        model = BinDetectorOpenVino(FRCNN_INCEPTION_OPENVINO.CFG_PATH,
                                    FRCNN_INCEPTION_OPENVINO.DETECTOR_PATH,
                                    classes=classes,
                                    threshold=info['THRESHOLD'],
                                    excluded_classes=list_difference(classes,
                                                                     ['Overflowed',
                                                                      'Bin']) if args.bin_only == 1 else [],
                                    box_area_limit=info['BOX_AREA_LIMIT'],
                                    model_version=info['MODEL_VERSION'],
                                    resize_h=800,
                                    keep_aspect_ratio=True
                                    )
        h = get_model_hash(FRCNN_INCEPTION_OPENVINO.DETECTOR_PATH)
        print("Loaded frcnn inception open vino model %s" % h)
    elif args.model == 'frcnn_inception_tf':
        from model.ssd_detector import BinDetectorSSD

        classes = parse_classes_file(FRCNN_INCEPTION_TF.CLASSES_NAMES)
        info = parse_info_file(FRCNN_INCEPTION_TF.INFO_PATH)
        model = BinDetectorSSD(FRCNN_INCEPTION_TF.DETECTOR_PATH,
                               classes=classes,
                               threshold=info['THRESHOLD'],
                               excluded_classes=list_difference(classes,
                                                                ['Overflowed', 'Bin']) if args.bin_only == 1 else [],
                               box_area_limit=info['BOX_AREA_LIMIT'],
                               model_version=info['MODEL_VERSION']
                               )
        h = get_model_hash(FRCNN_INCEPTION_TF.DETECTOR_PATH)
        print("Loaded frcnn inception tf model %s" % h)
    elif args.model == 'frcnn_resnet_tf':
        from model.ssd_detector import BinDetectorSSD

        classes = parse_classes_file(FRCNN_RESNET_TF.CLASSES_NAMES)
        info = parse_info_file(FRCNN_RESNET_TF.INFO_PATH)
        model = BinDetectorSSD(FRCNN_RESNET_TF.DETECTOR_PATH,
                               classes=classes,
                               threshold=info['THRESHOLD'],
                               excluded_classes=list_difference(classes,
                                                                ['Overflowed', 'Bin']) if args.bin_only == 1 else [],
                               box_area_limit=info['BOX_AREA_LIMIT'],
                               model_version=info['MODEL_VERSION']
                               )
        h = get_model_hash(FRCNN_RESNET_TF.DETECTOR_PATH)
        print("Loaded ssd fpn resnet model %s" % h)
    else:
        raise ValueError("INVALID MODEL")

    print(info)
    print("Bifocal %s" % (args.bifocal == 1))

    with open('/tmp/eval.txt', 'a') as w:
        print('Model version %s; %s; threshold is %.2f' % (
            info['MODEL_VERSION'], ('bifocal' if args.bifocal == 1 else 'no bifocal'), info['THRESHOLD']),
              file=w)
        print('Detector sha1 %s' % h, file=w)
        w.flush()

    predictions_path = PREDICTIONS_PATH + "_" + args.model + "_" + str(args.bifocal)
    if os.path.exists(predictions_path):
        shutil.rmtree(predictions_path)
    os.makedirs(predictions_path)

    sample_times = []

    for i, fname in enumerate(sorted(glob.glob(os.path.join(TEST_IMAGES_PATH, '*.jpg')))):
        name = os.path.basename(fname).replace('.jpg', '.txt')
        if not os.path.exists(os.path.join(TEST_GROUND_TRUTH_PATH, name)):
            print("Skip %s" % fname)
            continue
        img = skimage.io.imread(fname)

        start_time = time()
        detections = model.predict_on_image(img, bifocal=(args.bifocal == 1))
        time_elapsed = time() - start_time
        if i > 0 and i % 50 == 0:
            print("{} in {:.3f}s".format(fname, time_elapsed))
            sample_times.append(time_elapsed)

        '''if detections:
            skimage.io.imsave(os.path.join(predictions_path, name + '.jpg'), res)
        elif gt_file_has_class(os.path.join(TEST_GROUND_TRUTH_PATH, name), 'bin_pos'):
            skimage.io.imsave(os.path.join(predictions_path, 'no_det_' + name + '.jpg'), res)'''

        with open(os.path.join(TEST_DETECTIONS_PATH, name), 'w') as w:
            for d in range(len(detections)):
                for dict in detections[d]:
                    print('{} {} {} {} {} {}'.format(LOOKUP.get(dict['class'].lower(), dict['class'].lower()), dict['score'],
                                                 *dict['box']), file=w)
            w.flush()

    t = float(np.mean(np.array(sample_times)))
    with open('/tmp/eval.txt', 'a') as w:
        print('Average inference time %.3f seconds' % t, file=w)
        w.flush()
