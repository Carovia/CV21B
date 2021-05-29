from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


# 加载数据集
data = load('cv21b-embeddings.npz')
train_x, train_y, test_x, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (train_x.shape[0], test_x.shape[0]))
# 归一化
in_encoder = Normalizer(norm='l2')
train_x = in_encoder.transform(train_x)
test_x = in_encoder.transform(test_x)
# 拟合模型
model = SVC(kernel='linear', probability=True)
model.fit(train_x, train_y)
# 预测
yhat_train = model.predict(train_x)
yhat_test = model.predict(test_x)
score_train = accuracy_score(train_y, yhat_train)
print('Accuracy: train=%.3f' % (score_train*100))

f = open('data/val_result.txt', 'w')
for i in range(len(test_y)):
    f.write('%s.jpg %s\n' % (test_y[i], yhat_test[i]))
f.close()
