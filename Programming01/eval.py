from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

width = 64
height = 64
channel = 3


def read_image(path):
    image = cv2.imread(path)
    # image = cv2.resize(image, (width, height))
    return img_to_array(image).reshape(width, height, channel).astype("float")


def load_train_data(train_path):
    data = []
    categories = []
    for category in os.listdir(train_path):
        category_path = os.path.join(train_path, category)
        for image in os.listdir(category_path):
            data.append(read_image(os.path.join(category_path, image)))
            categories.append(category)
    train_data = np.array(data) / 255.0
    train_label = np.array(to_categorical(categories))
    print("finish load train data")
    return train_data, train_label


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(height, width, channel)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=80, activation='softmax'))
    return model


def train(x_train, y_train, x_val, y_val, model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=30, batch_size=256, verbose=2)


def load_res(res_path):
    with open(res_path, 'r') as f:
        items = f.readlines()
        items = [item.strip().split() for item in items]
    iid_to_cid = {item[0]: item[1] for item in items}
    return iid_to_cid


def cal_acc(anno, pred):
    sample_num = len(anno)
    hit_cnt = 0.0
    for iid, cid in anno.items():
        if iid in pred and cid == pred[iid]:
            hit_cnt += 1
    return hit_cnt / sample_num


def load_val_data(val_path, res_path):
    res = load_res(res_path)
    val = []
    categories = []
    for item in res:
        val.append(read_image(os.path.join(val_path, item)))
        categories.append(int(res[item]))
    val_data = np.array(val) / 255.0
    val_label = np.array(to_categorical(categories))
    return val_data, val_label


train_data, train_label = load_train_data("train")
val_data, val_label = load_val_data('val', 'val_anno.txt')
my_model = build_model()
train(train_data, train_label, val_data, val_label, my_model)
