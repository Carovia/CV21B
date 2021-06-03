import os
import shutil
from keras.preprocessing.image import ImageDataGenerator


def make_dir(path):
    for image in os.listdir(path):
        dir_path = path + '/' + image[:-4]
        os.makedirs(dir_path)
        shutil.move(path + '/' + image, dir_path + '/' + image)


def augment_image(data_path):
    generator = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    category_generator = generator.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=50,
        save_to_dir=data_path,
        save_format='jpg')

    for i in range(5):
        category_generator.next()


# make_dir('data/test')
augment_image('data/gallery')