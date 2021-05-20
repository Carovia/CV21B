import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from ssd.structures.container import Container


class VOCDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'lid', 'handle', 'oven', 'door', 'faucet', 'picture', 'stove', 'cabinet', 'bowl', 'sink', 'counter', 
                   'kitchen', 'cabinets', 'wall', 'handles', 'rack', 'shelf', 'towel', 'dish', 'floor', 'tiles', 'cup', 'bottle', 
                   'spoon', 'knob', 'pot', 'doors', 'light', 'ceiling', 'window', 'drawer', 'fruit', 'chairs', 'lights', 'chair', 
                   'table', 'microwave', 'carpet', 'wheel', 'stripes', 'basket', 'bathroom', 'container', 'mirror', 'toilet', 'curtain', 'seat', 
                   'flowers', 'sauce', 'pan', 'flower', 'leaves', 'food', 'computer', 'desk', 'monitor', 'box', 'leg', 'keyboard', 
                   'wires', 'phone', 'frame', 'mug', 'television', 'suitcase', 'tv', 'stand', 'paper', 'remote', 'sign', 'shirt', 
                   'bed', 'blanket', 'room', 'outlet', 'white', 'side', 'plate', 'tire', 'tile', 'ground', 'man', 'lines', 
                   'sky', 'tree', 'roof', 'building', 'helmet', 'pants', 'foot', 'concrete', 'house', 'street', 'hair', 'bike', 
                   'vehicle', 'person', 'motorcycle', 'headlight', 'dog', 'pillows', 'books', 'couch', 'corner', 'sofa', 'speaker', 'legs', 
                   'head', 'paw', 'arm', 'cushion', 'jacket', 'coat', 'ski pole', 'a', 'ski', 'bottom', 'ring', 'letter', 
                   'snow', 'tag', 'writing', 'skis', 'poles', 'sticker', 'pole', 'tray', 'bench', 'book', 'pillow', 'refrigerator', 
                   'cheese', 'bar', 'duck', 'water', 'beak', 'eyes', 'rug', 'plant', 'bag', 'cord', 'cloth', 'line', 
                   'this', 'boat', 'bicycle', 'sand', 'the', 'red', 'black', 'beach', 'rocks', 'rock', 'glasses', 'finger', 
                   'woman', 'shoe', 'top', 'sunglasses', 'belt', 'umbrella', 'dress', 'shoes', 'skirt', 'purse', 'trim', 'tip', 
                   'lady', 'part', 'shadow', 'shade', 'hand', 'knife', 'blinds', 'board', 'edge', 'mountain', 'snowboard', 'stick', 
                   'weeds', 'background', 'hill', 'bear', 'crack', 'hole', 'teddy bear', 'street light', 'truck', 'cloud', 'plants', 'plane', 
                   'air', 'road', 'branches', 'fence', 'distance', 'clouds', 'tail', 'logo', 'dirt', 'buildings', 'tower', 'car', 
                   'wing', 'statue', 'grass', 'path', 'collar', 'body', 'reflection', 'glass', 'clock', 'holder', 'wood', 'back', 
                   'jeans', 'sweater', 'strap', 'buttons', 'signs', 'hood', 'cover', 'pipe', 'windows', 'branch', 'area', 'brick', 
                   'steps', 'numbers', 'post', 'pocket', 'backpack', 'luggage', 'metal', 'bolt', 'cellphone', 'painting', 'eye', 'sock', 
                   'bat', 'object', 'lamp', 'laptop', 'button', 'screen', 'fur', 'jar', 'vase', 'words', 'trash can', 'vegetables', 
                   'bridge', 'candle', 'wire', 'rope', 'cart', 'train car', 'train', 'chain', 'paint', 'guy', 'watch', 'wrist', 
                   'men', 't-shirt', 'boy', 'shorts', 'ear', 'people', 'knee', 'bus', 'blue', 'stripe', 'onion', 'bucket', 
                   'hands', 'curtains', 'cap', 'sidewalk', 'flag', 'pavement', 'license plate', 'ball', 'racket', 'rail', 'stem', 'leaf', 
                   'headlights', 'photo', 'number', 'letters', 'curb', 'cake', 'hat', 'front', 'patch', 'elephant', 'trunk', 'feet', 
                   'cat', 'whiskers', 'face', 'nose', 'windshield', 'wheels', 'banner', 'city', 'neck', 'tie', 'clothes', 'ripples', 
                   'bricks', 'sun', 'skier', 'trees', 'animal', 'surface', 'birds', 'ocean', 'mouth', 'gate', 'bracelet', 'player', 
                   'court', 'spectator', 'tennis racket', 'circle', 'chimney', 'structure', 'shore', 'river', 'stone', 'girl', 'horse', 'sneakers', 
                   'jet', 'runway', 'field', 'airplane', 'skateboard', 'sneaker', 'surfboard', 'controller', 'bird', 'engine', 'cone', 'stairs', 
                   'ramp', 'skateboarder', 'zebra', 'mane', 'hoof', 'poster', 'sleeve', 'bushes', 'kite', 'child', 'park', 'cow', 
                   'cows', 'socks', 'horn', 'name', 'wave', 'waves', 'van', 'tablecloth', 'giraffe', 'design', 'vest', 'lettuce', 
                   'fork', 'napkin', 'meat', 'sandwich', 'bun', 'wine', 'cell phone', 'banana', 'tomato', 'bananas', 'pizza', 'tent', 
                   'walkway', 'word', 'can', 'carrot', 'spots', 'lettering', 'orange', 'slice', 'bread', 'baby', 'suit', 'broccoli', 
                   'kid', 'beard', 'pepper', 'crust', 'label', 'net', 'scarf', 'pen', 'shoulder', 'pattern', 'vegetable', 'column', 
                   'base', 'platform', 'hot dog', 'spot', 'batter', 'glove', 'catcher', 'umpire', 'jersey', 'baseball', 'uniform', 'gloves', 
                   'band', 'camera', 'tennis court', 'boots', 'string', 'railing', 'ears', 'tennis ball', 'log', 'boot', 'track', 'shadows', 
                   'frisbee', 'arms', 'panel', 'graffiti', 'elephants', 'tracks', 'horses', 'cars', 'apple', 'bush', 'pillar', 'mouse', 
                   'goggles', 'key', 'tree trunk', 'doorway', 'star', 'street sign', 'traffic light', 'boats', 'mountains', 'stop sign', 'balcony', 'awning', 
                   'fire hydrant', 'hydrant', 'arrow', 'train tracks', 'surfer', 'giraffes', 'palm tree', 'gravel', 'wetsuit', 'horns', 'zebras', 'sheep', 
                   'necklace', 'tusk', 'square', 'donut')

    def __init__(self, data_dir, split, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
