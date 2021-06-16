import numpy as np
import cv2
from mySOT.tracker.features.fhog import get_feature_maps, normalize_and_truncate, pca_feature_maps
from mySOT.tracker.features.table import TableFeature


def cos_window(size):
    cos_window = np.hanning(int(size[1]))[:, np.newaxis].dot(np.hanning(int(size[0]))[np.newaxis, :])
    return cos_window


def gaussian2d_labels(size, sigma):
    w, h = size
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    center_x, center_y = w / 2, h / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    return labels


def gaussian2d_rolled_labels(size, sigma):
    w, h = size
    xs, ys = np.meshgrid(np.arange(w)-w//2, np.arange(h)-h//2)
    dist = (xs**2+ys**2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    labels = np.roll(labels, -int(np.floor(size[0] / 2)), axis=1)
    labels = np.roll(labels, -int(np.floor(size[1] / 2)), axis=0)
    return labels


def fft2(x):
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)


def ifft2(x):
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)


def extract_hog_feature(img, cell_size=4):
    mapp = {'size_x': 0, 'size_y': 0, 'num_features': 0, 'map': 0}
    mapp = get_feature_maps(img, cell_size, mapp)
    mapp = normalize_and_truncate(mapp, 0.2)
    mapp = pca_feature_maps(mapp)
    features_map = mapp['map'].reshape((mapp['size_y'], mapp['size_x'], mapp['num_features']))
    features_map = np.insert(features_map, (0, -1), [0], axis=0)
    features_map = np.insert(features_map, (0, -1), [0], axis=1)
    return features_map


def extract_cn_feature(img,cell_size=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 - 0.5
    cn = TableFeature(fname='cn', cell_size=cell_size, compressed_dim=11, table_name="CNnorm",
                      use_for_color=True)

    if np.all(img[:, :, 0] == img[:, :, 1]):
        img = img[:, :, :1]
    else:
        # # pyECO using RGB format
        img = img[:, :, ::-1]
    h,w=img.shape[:2]
    cn_feature = \
    cn.get_features(img, np.array(np.array([h/2,w/2]), dtype=np.int16), np.array([h,w]), 1, normalization=False)[
        0][:, :, :, 0]
    # print('cn_feature.shape:', cn_feature.shape)
    # print('cnfeature:',cn_feature.shape,cn_feature.min(),cn_feature.max())
    gray = cv2.resize(gray, (cn_feature.shape[1], cn_feature.shape[0]))[:, :, np.newaxis]
    out = np.concatenate((gray, cn_feature), axis=2)
    return out