
from collections import namedtuple
import json
import numpy as np
from pathlib import Path
Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])


def load_calibration(root_dir, camera_name, sequence):
    filepath = Path(root_dir) / 'dataset' / 'calibrations' / f'sequence_0{sequence}' /f'{camera_name}.json'
    with open(filepath, 'r') as f:
        calib_dict = json.load(f)
    K = np.array(calib_dict.get("K")) if calib_dict.get("K") is not None else None
    R = np.array(calib_dict.get("R")) if calib_dict.get("R") is not None else None
    T = np.array(calib_dict.get("T")) if calib_dict.get("T") is not None else None
    dist = np.array(calib_dict.get("dist")) if calib_dict.get("dist") is not None else None
    view_id = camera_name
    return Calibration(K, R, T, dist, view_id)