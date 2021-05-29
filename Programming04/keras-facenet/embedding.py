from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
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


# 加载数据集
data = load('cv21b-dataset.npz')
train_x, train_y, test_x, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
# 加载 facenet 模型
model = load_model('facenet_keras.h5')
print('Loaded Model')
# 将训练集中每一张脸转化为向量
new_train_x = list()
for pixels in train_x:
    embedding = get_embedding(model, pixels)
    new_train_x.append(embedding)
new_train_x = asarray(new_train_x)
print(new_train_x.shape)
# 将测试（验证）集中每一张脸转化为向量
new_test_x = list()
for pixels in test_x:
    embedding = get_embedding(model, pixels)
    new_test_x.append(embedding)
new_test_x = asarray(new_test_x)
print(new_test_x.shape)
savez_compressed('cv21b-embeddings.npz', new_train_x, train_y, new_test_x, test_y)
