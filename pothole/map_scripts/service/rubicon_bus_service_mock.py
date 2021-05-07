import glob
import logging
import os
import shutil
import skimage.io
import skimage.transform

from settings import MOCK_INPUT_PATH
from utils import split_mov_into_frames

logger = logging.getLogger(__name__)


class MovIterator:
    SAVE_TO = '/tmp/footage'

    def __init__(self, items, fps=2, t=0):
        """
        Frame iterator that parses MOV files
        :param items: list of items (MOV files + jpg)
        :param fps: frames pers second rate when parsing MOV file
        :param t: wait period in seconds (frames to skip: t * fps)
        """
        self.items = items
        self.frames = iter([])
        self.index = -1
        self.fps = fps
        self.t = t
        self.to_skip = fps * t

    def __iter__(self):
        return self

    def __next__(self):

        frame = next(self.frames, None)
        while self.to_skip > 0:
            frame = next(self.frames, None)
            self.to_skip -= 1
        if frame is None:
            self.index += 1
            if self.index >= len(self.items):
                raise StopIteration
            current_item = self.items[self.index]
            if current_item.lower().endswith('.jpg'):
                self.frames = iter([current_item])
            else:
                logger.info("Processing %s" % current_item)
                if os.path.exists(MovIterator.SAVE_TO):
                    shutil.rmtree(MovIterator.SAVE_TO)
                os.makedirs(MovIterator.SAVE_TO)
                split_mov_into_frames(current_item, MovIterator.SAVE_TO, fps=self.fps)
                self.frames = iter(sorted(glob.glob(os.path.join(MovIterator.SAVE_TO, '*.jpg'))))

            return self.__next__()
        else:
            logger.info("Inference on %s" % frame)
            frame = skimage.io.imread(frame)

        self.to_skip = self.fps * self.t
        return frame


class RubiconBusServiceMock:

    def __init__(self, items, t=0):
        self.items = items
        self.t = t
        self.mov_iterator = None

    def fetch_data(self):
        data = {'lat': 0, 'long': 0, 'is_valid_gps': 0, 'speed': 0, 'track': 0, 'image': None, 'frame_rate': 2}
        if self.items is None:
            file = glob.glob(os.path.join(MOCK_INPUT_PATH, "*.jpeg"))[0]
            logger.info("Inference on %s" % file)
            img = skimage.io.imread(file)
        else:
            if self.mov_iterator is None:
                self.mov_iterator = MovIterator(self.items, t=self.t)
            img = next(self.mov_iterator)

        data['image'] = img
        return data
