from keras.models import load_model, Sequential, Model
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import keras.backend as K
import tensorflow as tf
from numpy import load, savez_compressed
import numpy as np


def merge():
    train0 = load('result/cv21b-train-0.npz')
    train0_x, train0_y = train0['arr_0'], train0['arr_1']
    train1 = load('result/cv21b-train-1.npz')
    train1_x, train1_y = train1['arr_0'], train1['arr_1']
    train2 = load('result/cv21b-train-2.npz')
    train2_x, train2_y = train2['arr_0'], train2['arr_1']
    train3 = load('result/cv21b-train-3.npz')
    train3_x, train3_y = train3['arr_0'], train3['arr_1']
    train4 = load('result/cv21b-train-4.npz')
    train4_x, train4_y = train4['arr_0'], train4['arr_1']
    train_x = train0_x
    train_x = np.append(train_x, train1_x, axis=0)
    train_x = np.append(train_x, train2_x, axis=0)
    train_x = np.append(train_x, train3_x, axis=0)
    train_x = np.append(train_x, train4_x, axis=0)
    train_y = train0_y
    train_y = np.append(train_y, train1_y, axis=0)
    train_y = np.append(train_y, train2_y, axis=0)
    train_y = np.append(train_y, train3_y, axis=0)
    train_y = np.append(train_y, train4_y, axis=0)
    print(train_x.shape)
    print(train_y.shape)
    savez_compressed('result/cv21b-train.npz', train_x, train_y)


def triplet_loss(alpha=0.2, batch_size=32):
    def _triplet_loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:batch_size], y_pred[batch_size:int(2 * batch_size)], y_pred[-batch_size:]

        pos_dist = K.sqrt(K.sum(K.square(anchor - positive), axis=-1))
        neg_dist = K.sqrt(K.sum(K.square(anchor - negative), axis=-1))

        basic_loss = pos_dist - neg_dist + alpha

        idxs = tf.where(basic_loss > 0)
        select_loss = tf.gather_nd(basic_loss, idxs)

        loss = K.sum(K.maximum(basic_loss, 0)) / tf.cast(tf.maximum(1, tf.shape(select_loss)[0]), tf.float32)
        return loss

    return _triplet_loss


def retrain(data_path='result/cv21b-train.npz'):
    # 加载预训练模型
    base_model = load_model('model/facenet_inception_resnet_v1.h5')
    # 加载训练数据
    data = load(data_path)
    data_x, data_y = data['arr_0'].astype('float32'), data['arr_1'].astype('int32')
    for i in range(len(data_x)):
        mean, std = data_x[i].mean(), data_x[i].std()
        data_x[i] = (data_x[i] - mean) / std
    print('Loaded Data:', data_x.shape, data_y.shape)
    # 根据类别标准化标签
    classes = 100
    data_y = np_utils.to_categorical(data_y, classes)
    # 定义训练参数
    batch_size = 32
    lr = 1e-4
    # 冻结部分层
    for layer in base_model.layers[:-16]:
        layer.trainable = False
    # 添加最后的分类层
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(100, activation='sigmoid')
    ])
    print(model.summary())
    # 编译并训练
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    model.fit(x=data_x, y=data_y, batch_size=batch_size, epochs=50, verbose=2,
              validation_split=0.2, shuffle=True)
    save_model = Sequential()
    for layer in model.layers[:-4]:
        save_model.add(layer)
    # 保存模型
    save_model.save('model/my_model.h5')


# merge()
retrain()
