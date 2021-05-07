import os, sys
import shutil
import cv2
import glob
import matplotlib.pyplot as plt

det1_folder = sys.argv[1]
det2_folder = sys.argv[2]
save_folder = sys.argv[3]
#save_folder = '/home/mcwi9/Model_Optimization/ml_nas/Rubicon/SOW2/pothole-retag-data/model_results/Rub-tag-T11/one_day_data/21_04_15/6k_11k_09t_plots/'

for filename in glob.glob(det1_folder + "*.jpg"):
    #print(filename)
    image1 = plt.imread(filename)
    img_name = filename.split('/')[-1]
    plot_name = img_name.split('.jpg')[0]
    print(img_name)
    fig = plt.figure(figsize=(20, 15))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title("11k 0.9t results")
    if os.path.exists(det2_folder + '/' + img_name):
        print("hello")
        image2 = plt.imread(det2_folder + '/' + img_name)
        fig.add_subplot(1, 2, 2)
        plt.imshow(image2)
        plt.axis('off')
        plt.title("6k 0.9t results")
    plt.axis('off')
    plt.savefig(save_folder + plot_name + '.png', bbox_inches='tight', pad_inches=0)
    #fig.savefig(save_folder + plot_name + '.png')

