import json
from pathlib import Path
import cv2
import numpy as np








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

