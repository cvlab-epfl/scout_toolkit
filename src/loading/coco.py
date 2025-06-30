import json

def load_coco(coco_path):
    with open(coco_path, 'r') as f:
        coco_dict = json.load(f)
    return coco_dict