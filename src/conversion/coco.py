"""
Export script to generate coco data format based on our json
"""

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import re
from itertools import chain

def extract_cam_number(cam_id: str) -> int:
    match = re.search(r'(\d+)$', cam_id)
    if match:
        return int(match.group(1))
    raise ValueError(f"No trailing number found in cam_id: {cam_id}")

def convert_bbox(bbox_in):
    (x1, y1), (x2, y2) = bbox_in
    width = x2 - x1
    height = y2 - y1
    return (x1, y1, width, height)

def create_annotation(annotation, frame_id, seq_id):
    track_id = annotation['track_id']
    return [{
        "id": f"{int(seq_id):01d}{int(extract_cam_number(cam_id)):02d}{int(frame_id):05d}{int(track_id):06d}",
        "image_id": f"{int(seq_id):01d}{int(extract_cam_number(cam_id)):02d}{int(frame_id):05d}",
        "category_id": track_id,
        "bbox": convert_bbox(annotation['projections_2d'][cam_id]),
        "world": [annotation["cuboid_3d"]['Xw'], annotation["cuboid_3d"]['Yw'], annotation["cuboid_3d"]['Zw']]
    } for cam_id in annotation['projections_2d'].keys()]

def create_image(frame, seq_id, image_width, image_height):
    if not frame['annotations']:
        return []

    return [{
        "file_name": f"images/{cam_id}/image_{int(frame['frame_id'])}.jpg",
        "height": image_height,
        "width": image_width,
        "date_captured": frame['timestamp_ms'] * 1000,
        "id": f"{int(seq_id):01d}{int(extract_cam_number(cam_id)):02d}{int(frame['frame_id']):05d}"
    } for cam_id in frame['annotations'][0]['projections_2d'].keys()]

def create_category(frame_annotation):
    category_dict = {
        "supercategory": "person",
        "id": frame_annotation['track_id'],
        "name": f"person_{frame_annotation['track_id']}"
    }

    return category_dict


def to_coco(input, seq_id, image_width, image_height) -> dict:
    """

    Generates a coco dict in the format as below and saves to output
     {
                "info": {
                    "description": "SCOUT 2025 Dataset",
                    "url": "http://scoutdataseturl",
                    "version": "1.0",
                    "year": 2025,
                    "contributor": "EPFL CVLab",
                    "date_created": "2025"
                },
                "images": [
                  ],
                "annotations": [
                        {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 1, 1, 1]
                        }
                ],
                "categories": [{
                "supercategory": "person",
                "id": 1,
                "name": "person"
                },
                ],
        "licenses": [{"url":None, "id": 1, "name": None}]
    }

    """
    return {
        "info": {
            "description": "SCOUT 2025 Dataset",
            "url": "http://scoutdataseturl",
            "version": "1.0",
            "year": 2025,
            "contributor": "EPFL CVLab",
            "date_created": "2025"
        },
        "images": list(chain.from_iterable(
            create_image(frame, seq_id, image_width, image_height)
            for frame in input['frames']
        )),
        "annotations": [
            ann
            for frame in input['frames']
            for raw_ann in frame["annotations"]
            for ann in create_annotation(raw_ann, frame['frame_id'], seq_id)
        ],
        "categories": [
            create_category(annotation)
            for frame in input['frames']
            for annotation in frame["annotations"]
        ]
    }

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_json', type=str, default='')
    parser.add_argument('--seq_id', type=int, default=1)
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--output', type=str, default='coco')

    args = parser.parse_args()
    seq_dir = Path(args.image_dir) / f'seq{args.seq_id}'
    # grab an image to determine image size
    first_image = None
    for cam_dir in seq_dir.iterdir():
        if cam_dir.is_dir():
            for image in cam_dir.iterdir():
                if image.suffix.lower() == ".jpg":
                    first_image = image
                    break
        if first_image:
            break

    with Image.open(first_image) as img:
        width, height = img.size

    # load the json
    import json

    with open(Path(args.input_json), 'r') as f:
        input_dict = json.load(f)

    with open(Path(args.output), 'w') as f:
        json.dump(to_coco(input_dict, args.seq_id, width, height), f)

if __name__=="__main__":
    main()
