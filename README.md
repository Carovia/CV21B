# CV21B

## Programming01 图像分类
- 将数据集解压放在 Programming01 目录下即可

## Programming02 目标检测
- 核心代码参考
  - https://github.com/lufficc/SSD.git
- 使用提示
  - 可以下载好模型后根据 Programming02/SSD/README.md 的提示运行 demo.py 文件
- 训练好的模型
  - 链接：https://pan.baidu.com/s/10Zxafm9_lvTviQ86gh6AVQ 
  - 提取码：m2d0

## Programming03 语义识别
- 核心代码参考
  - https://github.com/jfzhang95/pytorch-deeplab-xception
- 使用说明
  - 参考：https://blog.csdn.net/yx868yx/article/details/113778713
  - 数据集需要按照 VOC 格式准备

## Programming04 人脸识别
- 实现参考
  - https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
- 使用提示
  - 下载预训练模型到 keras-facenet/model 目录下
    - 链接：https://pan.baidu.com/s/1MqvStlnv2-yhAIkqH7gtLQ
    - 提取码：h808
  - 图像数据解压到 keras-facenet/data 目录下，可以使用 prepare.py 将图片存放到同名文件夹下
  - 依次执行 extractor.py embedding.py classify.py 文件即可
  - 结果文件在 keras-facenet/result 目录