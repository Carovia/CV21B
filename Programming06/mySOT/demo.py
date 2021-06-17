import os
import shutil
import zipfile
import cv2
from mySOT.tracker.kcf import KCF
from mySOT.tracker.mosse import MOSSE
from mySOT.tracker.cn import CN


def track(image_dir, tracker, res_path):
    gt = open(os.path.join(image_dir, 'groundtruth.txt'), 'r')
    bbox = gt.readline()[:-1].split(',')
    gt.close()
    x_min = min(float(bbox[0]), float(bbox[2]), float(bbox[4]), float(bbox[6]))
    y_min = min(float(bbox[1]), float(bbox[3]), float(bbox[5]), float(bbox[7]))
    x_max = max(float(bbox[0]), float(bbox[2]), float(bbox[4]), float(bbox[6]))
    y_max = max(float(bbox[1]), float(bbox[3]), float(bbox[5]), float(bbox[7]))
    roi = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    frame = cv2.imread(os.path.join(image_dir, os.listdir(image_dir)[0]))
    # print(roi)
    # print(frame.shape)

    tracker.init(frame, roi)
    f = open(res_path, 'w')
    f.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7]))

    for image in os.listdir(image_dir)[1:-1]:
        frame = cv2.imread(os.path.join(image_dir, image))
        x, y, w, h = tracker.update(frame)
        print(image, x, y, w, h)
        f.write('%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (x, y, x+w, y, x+w, y+h, x, y+h))
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 1)
        cv2.imshow('tracking', frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    f.close()


def train(data_dir, res_dir, tracker):
    gt_dir = os.path.join(res_dir, 'groundtruth')
    test_dir = os.path.join(res_dir, 'test')
    if os.path.exists(gt_dir):
        shutil.rmtree(gt_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(gt_dir)
    os.mkdir(test_dir)

    for video in os.listdir(data_dir):
        img_dir = os.path.join(data_dir, video)
        res_file = os.path.join(test_dir, video + '.txt')
        track(img_dir, tracker, res_file)
        shutil.copy(os.path.join(img_dir, 'groundtruth.txt'), os.path.join(gt_dir, video + '.txt'))

    zip_gt = zipfile.ZipFile(os.path.join(res_dir, 'gt.zip'), 'w')
    for file in os.listdir(gt_dir):
        zip_gt.write(os.path.join(gt_dir, file))
    zip_gt.close()

    zip_test = zipfile.ZipFile(os.path.join(res_dir, 'test.zip'), 'w')
    for file in os.listdir(test_dir):
        zip_test.write(os.path.join(test_dir, file))
    zip_test.close()


def test(data_dir, res_dir, tracker):
    test_dir = os.path.join(res_dir, 'test')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)

    for video in os.listdir(data_dir):
        img_dir = os.path.join(data_dir, video)
        res_file = os.path.join(test_dir, video + '.txt')
        track(img_dir, tracker, res_file)


if __name__ == '__main__':
    train_dir = '../data/trainval'
    test_dir = '../data/test_public'
    res_path = '../result'
    myTracker = KCF()
    # myTracker = MOSSE()
    # myTracker = CN()

    # train(train_dir, res_path, myTracker)
    # test(test_dir, res_path, myTracker)
