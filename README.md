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

## Programming05 光学字符识别

## Programming06 单目标跟踪
- 实现参考
  - https://github.com/fengyang95/pyCFTrackers
  - https://github.com/uoip/KCFpy
  - https://github.com/chuanqi305/KCF
- 数据下载
  - 链接：https://pan.baidu.com/s/1S1gZqqC197NLlq5WO3OvPQ
  - 提取码：ax3q
- 运行方法
  - 下载代码后进入Programming06的目录，将图像数据解压存放到data目录下
  - 运行mySOT/demo.py可以在result目录下生成相应的预测文件
    - train方法用于预测trainval中的文件，生成结果并打包
    - test方法用于预测test_public中的文件，生成结果
    - 可以根据需要修改路径参数，例如tracker、图像路径等
  - 运行eval.py可以查看准确性
    - result文件夹下有使用不同算法的已经打包的结果集，将其复制到上一级目录（result/）下即可直接运行eval.py