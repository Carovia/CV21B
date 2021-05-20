import math
import os
import json
from xml.dom.minidom import Document
import random


def count_categories(json_path, res_path):
    with open(json_path, 'r') as f_json:
        json_data = json.load(f_json)
        categories = {}
        for image_name in json_data:
            objects = json_data[image_name]['objects']
            for object in objects:
                categories[objects[object]['category']] = 0
        f = open(res_path, 'w')
        count = 0
        for category in categories:
            count = count + 1
            if count == 12:
                f.write('\n')
                count = 0
            f.write('\'' + category + '\', ')
        f.close()
    f_json.close()


def create_xml(json_path, xml_dir):
    with open(json_path, 'r') as f_json:
        json_data = json.load(f_json)
        for image_name in json_data:
            f = open(xml_dir + '/' + image_name[:-4] + '.xml', 'w')
            f.write(Document().toprettyxml(indent="  "))
            f.writelines('<annotation>\n')
            f.writelines('  <folder>JPEGImages</folder>\n')
            f.writelines('  <filename>%s</filename>\n' % image_name)
            f.writelines('  <source>\n')
            f.writelines('    <database>Unknown</database>\n')
            f.writelines('  </source>\n')
            f.writelines('  <size>\n')
            f.writelines('    <width>%d</width>\n' % json_data[image_name]['width'])
            f.writelines('    <height>%d</height>\n' % json_data[image_name]['height'])
            f.writelines('    <depth>%d</depth>\n' % json_data[image_name]['depth'])
            f.writelines('  </size>\n')
            f.writelines('  <segmented>0</segmented>\n')
            objects = json_data[image_name]['objects']
            for object in objects:
                f.writelines('  <object>\n')
                f.writelines('    <name>%s</name>\n' % objects[object]['category'])
                f.writelines('    <pose>Unspecified</pose>\n')
                f.writelines('    <truncated>0</truncated>\n')
                f.writelines('    <difficult>0</difficult>\n')
                f.writelines('    <bndbox>\n')
                f.writelines('      <xmin>%d</xmin>\n' % objects[object]['bbox'][0])
                f.writelines('      <ymin>%d</ymin>\n' % objects[object]['bbox'][1])
                f.writelines('      <xmax>%d</xmax>\n' % objects[object]['bbox'][2])
                f.writelines('      <ymax>%d</ymax>\n' % objects[object]['bbox'][3])
                f.writelines('    </bndbox>\n')
                f.writelines('  </object>\n')
            f.writelines('</annotation>')
            f.close()
            print('finish %s' % image_name)
    f_json.close()


def create_txt(image_dir, txt_path):
    f = open(txt_path, 'w')
    for image in os.listdir(image_dir):
        f.writelines(image[:-4] + '\n')
    f.close()


def random_split(src_path, res1_path, res2_path):
    src = open(src_path, 'r')
    res1 = open(res1_path, 'w')
    res2 = open(res2_path, 'w')
    total = src.readlines()
    sum1 = math.floor(len(total) * 0.8)
    sum2 = len(total) - sum1
    train = random.sample(total, sum1)
    val = random.sample(total, sum2)
    for i in train:
        res1.writelines(i)
    for i in val:
        res2.writelines(i)


# count_categories('train/train.json', 'categories.txt')
# create_xml('train/train.json', 'datasets/Annotations')
# create_xml('val/val.json', 'datasets/Annotations')
# create_txt('train/train', 'datasets/ImageSets/Main/trainval.txt')
# create_txt('val/val', 'datasets/ImageSets/Main/test.txt')
random_split('datasets/ImageSets/Main/trainval.txt', 'datasets/ImageSets/Main/train.txt', 'datasets/ImageSets/Main/val.txt')