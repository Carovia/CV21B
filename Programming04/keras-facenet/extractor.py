from PIL import Image
from numpy import asarray, savez_compressed
from mtcnn.mtcnn import MTCNN
import os


# 从给定的图像中提取人脸
def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    # 转换成数组
    pixels = asarray(image)
    # 使用默认权重的检测器
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    # 从第一张脸提取边界框
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # 确定脸的位置并提取
    face = pixels[y1:y2, x1:x2]
    # 调整为指定大小
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# 从类别中加载图像
def load_faces(directory):
    faces = list()
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        face = extract_face(path)
        faces.append(face)
    return faces


# 加载数据集
def load_dataset(directory):
    x, y = list(), list()
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        x.extend(faces)
        y.extend(labels)
    return asarray(x), asarray(y)


train_x, train_y = load_dataset('data/gallery')
print(train_x.shape, train_y.shape)
test_x, test_y = load_dataset('data/val')
print(test_x.shape, test_y.shape)
savez_compressed('cv21b-dataset.npz', train_x, train_y, test_x, test_y)