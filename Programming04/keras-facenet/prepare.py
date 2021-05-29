import os
import shutil


def make_dir(path):
    for image in os.listdir(path):
        dir_path = path + '/' + image[:-4]
        os.makedirs(dir_path)
        shutil.move(path + '/' + image, dir_path + '/' + image)
