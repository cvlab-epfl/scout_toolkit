import json
from pathlib import Path
import cv2
import numpy as np


def load_coco(coco_path):
    with open(coco_path, 'r') as f:
        coco_dict = json.load(f)
    return coco_dict

def parse_mot_line(line: str) -> dict:
    fields = line.strip().split(',')
    if len(fields) != 10:
        raise ValueError(f"Expected 10 comma-separated values, got {len(fields)}: {line}")

    return {
        "frame_id": int(fields[0]),
        "track_id": int(fields[1]),
        "bbox": {
            "x": float(fields[2]),
            "y": float(fields[3]),
            "w": float(fields[4]),
            "h": float(fields[5]),
        },
        "world": {
            "Xw": float(fields[7]),
            "Yw": float(fields[8]),
            "Zw": float(fields[9]),
        }
    }

def load_mot(mot_path, cam_name):
    filepath = Path(mot_path) / f"{cam_name}.txt"
    with open(filepath, 'r') as f:
        parsed_lines = [parse_mot_line(line) for line in f if line.strip()]
    return parsed_lines

def parse_individual(line: str) -> dict:
    fields = line.strip().split(',')
    if len(fields) != 9:
        raise ValueError(f"Expected 10 comma-separated values, got {len(fields)}: {line}")

    return {
        "track_id": int(fields[0]),
        "bbox": {
            "x": float(fields[1]),
            "y": float(fields[2]),
            "w": float(fields[3]),
            "h": float(fields[4]),
        },
        "world": {
            "Xw": float(fields[5]),
            "Yw": float(fields[6]),
            "Zw": float(fields[7]),
        }
    }

def load_individual(filepath, cam_name, frame_id):
    filepath = Path(filepath) / cam_name / f"image_{frame_id}.txt"
    with open(filepath, 'r') as f:
        parsed_lines = [parse_individual(line) for line in f if line.strip()]
    return parsed_lines


def get_annotations_for_frame_and_camera(data_dict, target_frame_id, target_cam_id):
    """
    Filters annotations from the full data dictionary for a specific frame and camera.

    Args:
        data_dict (dict): The COCO-style dictionary.
        target_frame_id (int): The frame ID to match.
        target_cam_id (str): The camera ID, e.g., "cvlabrpi23".

    Returns:
        list[dict]: List of annotation dicts matching the frame and camera.
    """
    cam_num = int(''.join(filter(str.isdigit, target_cam_id)))  # extract_cam_number
    matching_image_id_prefix = f"{int(data_dict['images'][0]['id'][0])}{cam_num:02d}{target_frame_id:05d}"

    return [
        ann for ann in data_dict["annotations"]
        if ann["image_id"] == matching_image_id_prefix
    ]

