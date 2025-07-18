from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import cv2
import functools
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple
import matplotlib.pyplot as plt

from .visualisation import draw_annotation
from .geometry import Calibration

def load_calibration(root_dir, camera_name, sequence):
    # New structure: calibrations are at root level, named cam_X.json
    # Convert camera_name (e.g., "cam_5") to cam number or use directly
    if camera_name.startswith('cam_'):
        cam_file = f'{camera_name}.json'
    else:
        # If old camera name format, try to extract number
        cam_num = int(''.join(filter(str.isdigit, camera_name)))
        cam_file = f'cam_{cam_num}.json'
    
    filepath = Path(root_dir) / 'calibrations' / cam_file
    with open(filepath, 'r') as f:
        calib_dict = json.load(f)
    K = np.array(calib_dict.get("K")) if calib_dict.get("K") is not None else None
    R = np.array(calib_dict.get("R")) if calib_dict.get("R") is not None else None
    T = np.array(calib_dict.get("T")) if calib_dict.get("T") is not None else None
    dist = np.array(calib_dict.get("dist")) if calib_dict.get("dist") is not None else None
    view_id = calib_dict.get("view_id", camera_name)  # Use view_id from file or fallback to camera_name
    return Calibration(K, R, T, dist, view_id)

def get_annotations_for_frame_and_camera(data_dict, target_frame_id, target_cam_id) -> list[dict]:
    """
    Filters annotations from the full data dictionary for a specific frame and camera.

    Args:
        data_dict (dict): The COCO-style dictionary.
        target_frame_id (int): The frame ID to match.
        target_cam_id (str): The camera ID, e.g., "cam_5".

    Returns:
        list[dict]: List of annotation dicts matching the frame and camera.
    """
    # Extract camera number from cam_X format
    if target_cam_id.startswith('cam_'):
        cam_num = int(target_cam_id.split('_')[1])
    else:
        cam_num = int(''.join(filter(str.isdigit, target_cam_id)))
    
    # Construct image ID based on new structure
    matching_image_id_prefix = f"{int(data_dict['images'][0]['id'][0])}{cam_num:02d}{target_frame_id:05d}"

    return [
        {'track_id':ann['category_id'], 'bbox':ann['bbox'], 'world':ann['world']}
        for ann in data_dict["annotations"]
        if ann["image_id"] == matching_image_id_prefix
    ]

def parse_individual(line: str) -> dict:
    fields = line.strip().split(',')
    if len(fields) != 9:
        raise ValueError(f"Expected 9 comma-separated values, got {len(fields)}: {line}")

    return {
        "track_id": int(fields[0]),
        "bbox": {
            "x": float(fields[1]),
            "y": float(fields[2]),
            "w": float(fields[3]),
            "h": float(fields[4]),
        },
        "world": {
            "Xw": float(fields[6]),
            "Yw": float(fields[7]),
            "Zw": float(fields[8]),
        }
    }


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

class Loader(ABC):
    def __init__(self, root_dir:str|Path, sequence:int = 1):
        self.root_dir = root_dir
        self.sequence = sequence
        # New structure: timestamps are at root level, named timestamps_sequence_X.json
        timestamp_path = Path(root_dir) / 'timestamps' / f'timestamps_sequence_{sequence}.json'
        with open(timestamp_path, 'r') as f:
            self.timestamp_dict = json.load(f)

    @abstractmethod
    def retrieve(self, target_frame_id, target_cam_id) -> list:
        raise NotImplementedError
    
    def retrieve_many(self, frame_ids: list[int], cam_ids: list[str]) -> dict:
        result = {}
        for cam_id in cam_ids:
            result[cam_id] = {}
            for frame_id in frame_ids:
                try:
                    result[cam_id][frame_id] = self.retrieve(frame_id, cam_id)
                except FileNotFoundError:
                    result[cam_id][frame_id] = []
        return result

    def get_frame(self, target_frame_id, target_cam_id):
        # New structure: images/sequence_X/cam_X/image_X.jpg
        imagedir = Path(self.root_dir) / 'images' / f'sequence_{self.sequence}'
        
        # Ensure camera name is in cam_X format
        if not target_cam_id.startswith('cam_'):
            cam_num = int(''.join(filter(str.isdigit, target_cam_id)))
            target_cam_id = f'cam_{cam_num}'
            
        imagepath = Path(imagedir) / target_cam_id / f"image_{target_frame_id}.jpg"

        # Load the image
        image = cv2.imread(str(imagepath))
        if image is None:
            raise FileNotFoundError(f"Could not load image at {imagepath}")

        return image
    
    def annotate_frame(self, target_frame_id, target_cam_id) -> np.ndarray:
        annotations = self.retrieve(target_frame_id=target_frame_id, target_cam_id=target_cam_id)
        image = self.get_frame(target_frame_id=target_frame_id, target_cam_id=target_cam_id)
        for ann in annotations:
            bbox = ann['bbox']
            if isinstance(bbox, dict):
                bbox = bbox['x'], bbox['y'], bbox['w'], bbox['h']

            image = draw_annotation(image, bbox, ann['track_id'], thickness = 2)
        return image

    def show_frame(self, image:np.ndarray) -> None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20, 20), dpi=300)
        plt.imshow(image_rgb)
        plt.axis('off')  # Hide axes
        plt.show()

    def get_timestamp(self, target_frame_id, target_cam_id) -> float:
        """
        Timestamp in seconds
        """
        # Ensure camera name is in cam_X format for timestamp lookup
        if not target_cam_id.startswith('cam_'):
            cam_num = int(''.join(filter(str.isdigit, target_cam_id)))
            target_cam_id = f'cam_{cam_num}'
            
        return self.timestamp_dict[str(target_frame_id)][target_cam_id]

    
class Coco(Loader):
    def load(self) -> dict:
        # New structure: annotations/coco_format/annotations_coco_sequence_X.json
        coco_path = Path(self.root_dir) / 'annotations' / 'coco_format' / f'annotations_coco_sequence_{self.sequence}.json'
        with open(coco_path, 'r') as f:
            self._annotations = json.load(f)
        return self._annotations

    def retrieve(self, target_frame_id, target_cam_id) -> list:
        if not hasattr(self, '_annotations'):
            self._annotations = self.load()

        return get_annotations_for_frame_and_camera(self._annotations, target_frame_id, target_cam_id)



class Individual(Loader):
    def __init__(self, root_dir: str | Path, sequence: int = 1, cache_size: int = 1000, image_resize=None):
        super().__init__(root_dir, sequence)
        self._frame_cache = {}
        self._max_cache = cache_size
        self._resize = image_resize  # tuple like (H, W)

    def get_frame(self, target_frame_id, target_cam_id):
        key = (target_frame_id, target_cam_id)
        if key in self._frame_cache:
            return self._frame_cache[key]

        # New structure: images/sequence_X/cam_X/image_X.jpg
        imagedir = Path(self.root_dir) / 'images' / f'sequence_{self.sequence}'
        
        # Ensure camera name is in cam_X format
        if not target_cam_id.startswith('cam_'):
            cam_num = int(''.join(filter(str.isdigit, target_cam_id)))
            target_cam_id = f'cam_{cam_num}'
            
        imagepath = Path(imagedir) / target_cam_id / f"image_{target_frame_id}.jpg"
        image = cv2.imread(str(imagepath))

        if image is None:
            raise FileNotFoundError(f"Could not load image at {imagepath}")

        if self._resize:
            image = cv2.resize(image, (self._resize[1], self._resize[0]))  # (W, H)

        # Store in cache
        if len(self._frame_cache) >= self._max_cache:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[key] = image

        return image
    
    def load(self, frame_id:int, cam_name:str) -> list:
        # New structure: annotations/individual_format/cam_X/image_X.txt
        individual_path = Path(self.root_dir) / 'annotations' / 'individual_format'
        
        # Ensure camera name is in cam_X format
        if not cam_name.startswith('cam_'):
            cam_num = int(''.join(filter(str.isdigit, cam_name)))
            cam_name = f'cam_{cam_num}'
            
        filepath = individual_path / cam_name / f"image_{frame_id}.txt"
        with open(filepath, 'r') as f:
            parsed_lines = [parse_individual(line) for line in f if line.strip()]
        return parsed_lines

    def retrieve(self, target_frame_id, target_cam_id) -> list:
        if not hasattr(self, '_annotations') or self._curr_cam != target_cam_id or self._curr_frame != target_frame_id:
            self._annotations = self.load(target_frame_id, target_cam_id)
            self._curr_cam = target_cam_id
            self._curr_frame = target_frame_id
        return self._annotations
    

class MOT(Loader):
    def load(self, cam_name:str) -> list:
        # New structure: annotations/mot_format/sequence_X/cam_X.txt
        mot_path = Path(self.root_dir) / 'annotations' / 'mot_format' / f'sequence_{self.sequence}'
        
        # Ensure camera name is in cam_X format
        if not cam_name.startswith('cam_'):
            cam_num = int(''.join(filter(str.isdigit, cam_name)))
            cam_name = f'cam_{cam_num}'
            
        filepath = mot_path / f"{cam_name}.txt"
        with open(filepath, 'r') as f:
            parsed_lines = [parse_mot_line(line) for line in f if line.strip()]
        return parsed_lines
    
    def retrieve(self, target_frame_id, target_cam_id) -> list:
        if not hasattr(self, '_annotations') or self._curr_cam != target_cam_id:
            self._annotations = self.load(target_cam_id)
            self._curr_cam = target_cam_id
            # Optional speedup: index by frame_id
            self._index_by_frame = {}
            for ann in self._annotations:
                self._index_by_frame.setdefault(ann["frame_id"], []).append(ann)
        return self._index_by_frame.get(target_frame_id, [])




class AnnotationDataset(Dataset):
    def __init__(self, loader: Individual, frame_ids: List[int], cam_ids: List[str]):
        self.loader = loader
        self.frame_cam_pairs = [(f, c) for c in cam_ids for f in frame_ids]

    def __len__(self):
        return len(self.frame_cam_pairs)

    def __getitem__(self, idx: int) -> dict:
        frame_id, cam_id = self.frame_cam_pairs[idx]
        image = self.loader.get_frame(frame_id, cam_id)
        annots = self.loader.retrieve(frame_id, cam_id)
        timestamps = torch.tensor(self.loader.get_timestamp(frame_id, cam_id))

        # Convert image to torch tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # HWC -> CHW
        # Convert annotations to torch tensors
        if len(annots) > 0:
            bboxes = torch.tensor(
                [[a['bbox']['x'], a['bbox']['y'], a['bbox']['w'], a['bbox']['h']] for a in annots],
                dtype=torch.float32
            )
            world = torch.tensor(
                [[a['world']['Xw'], a['world']['Yw'], a['world']['Zw']] for a in annots],
                dtype=torch.float32
            )
            track_ids = torch.tensor([a['track_id'] for a in annots], dtype=torch.long)
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            world = torch.empty((0, 3), dtype=torch.float32)
            track_ids = torch.empty((0,), dtype=torch.long)


        return {
            'frame_id': frame_id,
            'cam_id': cam_id,
            'image': image_tensor,
            'bboxes': bboxes,
            'world': world,
            'track_ids': track_ids,
            'timestamps':timestamps
        }


def collate_fn(batch):
    images = [item['image'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]
    cam_ids = [item['cam_id'] for item in batch]
    bboxes = [item['bboxes'] for item in batch]
    world = [item['world'] for item in batch]
    track_ids = [item['track_ids'] for item in batch]
    timestamps = [item['timestamps'] for item in batch]

    return {
        'frame_ids': frame_ids,
        'cam_ids': cam_ids,
        'images': torch.stack(images),
        'bboxes': bboxes,        # List[Tensor]
        'world': world,          # List[Tensor]
        'track_ids': track_ids,  # List[Tensor]
        'timestamps' : timestamps
    }

