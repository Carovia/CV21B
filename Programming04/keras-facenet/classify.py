from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC, LinearSVC, NuSVC


# 加载数据集
train_data = load('result/cv21b-gallery-embedded.npz')
train_x, train_y = train_data['arr_0'], train_data['arr_1']
print('Loaded Train Data:', train_x.shape[0])
# 归一化
in_encoder = Normalizer(norm='l2')
train_x = in_encoder.transform(train_x)
# 拟合模型
model = SVC(kernel='linear', probability=True)  # 0.9657
# model = LinearSVC()  # 0.9600
# model = NuSVC()  # 0.9657
model.fit(train_x, train_y)
# 查看训练集的预测结果
yhat_train = model.predict(train_x)
score_train = accuracy_score(train_y, yhat_train)
print('Train Accuracy: %.3f' % (score_train * 100))


# 预测
def predict(data_path, model, res_path):
    data = load(data_path)
    data_x, data_y = data['arr_0'], data['arr_1']
    print('Loaded Data:', data_x.shape[0])
    # 归一化
    encoder = Normalizer(norm='l2')
    data_x = encoder.transform(data_x)
    yhat = model.predict(data_x)
    # 写入文件
    f = open(res_path, 'w')
    for i in range(len(data_y)):
        f.write('%s.jpg %s\n' % (data_y[i], yhat[i]))
    f.close()


predict('result/cv21b-val-embedded.npz', model, 'result/val_result.txt')
# predict('result/cv21b-test-embedded.npz', model, 'result/test_result.txt')
