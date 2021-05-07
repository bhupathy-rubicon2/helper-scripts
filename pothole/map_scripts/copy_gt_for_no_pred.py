import os
import sys
import glob
import shutil

from config import ROOT_DIR, TEST_GT_IMAGES_PATH, ITER

src_dir = TEST_GT_IMAGES_PATH
out_dir = ROOT_DIR + "/out-viz-" + ITER + "/all/"
dst_dir = ROOT_DIR + "/out-viz-" + ITER + "/fn/"


i = 0
for filename in glob.glob(src_dir+"/*.jpg"):
    if not os.path.exists(out_dir + os.path.basename(filename)):
        shutil.copy(filename, dst_dir + "/" + os.path.basename(filename))
        i += 1

print("files copied : ", i)

