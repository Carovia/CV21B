from numpy import load, expand_dims, asarray, savez_compressed
from keras.models import load_model


# 对图像获取人脸嵌入
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    # 标准化
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # 扩展维数，将数组转化为样本
    samples = expand_dims(face_pixels, axis=0)
    # 预测以获取向量
    yhat = model.predict(samples)
    return yhat[0]


# 加载数据并转化为向量
def transform(data_path, data_type, model):
    data = load(data_path)
    data_x, data_y = data['arr_0'], data['arr_1']
    print('Loaded Data:', data_x.shape, data_y.shape)
    new_data_x = list()
    for pixels in data_x:
        embedding = get_embedding(model, pixels)
        new_data_x.append(embedding)
    new_data_x = asarray(new_data_x)
    print('Embedded Data:', new_data_x.shape)
    save_name = 'result/cv21b-' + data_type + '-embedded.npz'
    savez_compressed(save_name, new_data_x, data_y)


# 加载 facenet 模型
my_model = load_model('model/facenet_inception_resnet_v1.h5')
print('Loaded Model')

# transform('result/cv21b-train.npz', 'train', my_model)
transform('result/cv21b-gallery.npz', 'gallery', my_model)
transform('result/cv21b-val.npz', 'val', my_model)
# transform('result/cv21b-test.npz', 'test', my_model)

# 16: 0.9539
# 99: 0.9087
# all: 0.8307
