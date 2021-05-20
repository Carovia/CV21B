import glob
import os
import time

import torch
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer


categories = [
    'background', 'lid', 'handle', 'oven', 'door', 'faucet', 'picture', 'stove', 'cabinet', 'bowl', 'sink', 'counter', 
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
    'necklace', 'tusk', 'square', 'donut'
]


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    
    objects = 0
    json_path = "demo/my_test.json"
    f = open(json_path, "w")
    f.write('{')
        
    model.eval()
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))
        
        f.write('"%s": {"height": %d, "width": %d, "depth": 3, "objects": {' % (image_name, height, width))
        for j in range(len(boxes)):
            objects = objects + 1
            f.write('"%d": {"category": "%s", "bbox": [%d, %d, %d, %d]}' % (objects, categories[labels[j]], boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]))
            if (j < len(boxes) - 1):
                f.write(', ')
                
        if i < len(image_paths) - 1:
            f.write('}}, ')
        else:
            f.write('}}')

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))
        
    f.write('}')
    f.close()


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
