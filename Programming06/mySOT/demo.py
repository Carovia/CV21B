import os
import cv2
from mySOT.tracker.kcf import KCF
from mySOT.tracker.mosse import MOSSE
from mySOT.tracker.cn import CN

if __name__ == '__main__':

    image_dir = '../data/trainval/zbdeuatx'
    # tracker = cv2.TrackerCSRT_create()
    tracker = KCF()
    # tracker = MOSSE()
    # tracker = CN()

    gt = open(os.path.join(image_dir, 'groundtruth.txt'), 'r')
    bbox = gt.readline()[:-1].split(',')
    gt.close()
    x_min = min(float(bbox[0]), float(bbox[2]), float(bbox[4]), float(bbox[6]))
    y_min = min(float(bbox[1]), float(bbox[3]), float(bbox[5]), float(bbox[7]))
    x_max = max(float(bbox[0]), float(bbox[2]), float(bbox[4]), float(bbox[6]))
    y_max = max(float(bbox[1]), float(bbox[3]), float(bbox[5]), float(bbox[7]))
    roi = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    frame = cv2.imread(os.path.join(image_dir, os.listdir(image_dir)[0]))
    print(roi)
    print(frame.shape)

    tracker.init(frame, roi)
    for image in os.listdir(image_dir)[1:-1]:
        frame = cv2.imread(os.path.join(image_dir, image))
        x, y, w, h = tracker.update(frame)
        print(image, x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imshow('tracking', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
