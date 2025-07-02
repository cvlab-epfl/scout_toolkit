"""
adjusts the offsets from the exported json to align with the frames generated here
"""

from argparse import ArgumentParser
from pathlib import Path
import json
from copy import deepcopy
import math


def trim_offset(input_dict, offsets_path) -> dict:
    """
    Amends frame indices based on offset file
    """
    offset_path = Path(offsets_path)
    assert offset_path.exists(), f"No dictionary at path: {offset_path}"
    with open(offset_path, 'r') as f:
        offset_dict = json.load(f)

    fixed_offset = int(math.ceil(max(offset_dict.values()) / 10.0)) * 10

    output_dict = deepcopy(input_dict)

    # Remove frames with frame_id < fixed_offset and adjust the rest
    output_dict['frames'] = [
        {
            **frame,
            'frame_id': frame['frame_id'] - fixed_offset
        }
        for frame in output_dict['frames']
        if frame['frame_id'] >= fixed_offset
    ]
    output_dict['total_frames'] = len(output_dict['frames'])
    return output_dict
    
def change_camera_names(input_dict: dict, camera_mapping: dict) -> dict:
    for frame in input_dict.get('frames', []):
        for annotation in frame.get('annotations', []):
            projections = annotation.get('projections_2d', {})
            annotation['projections_2d'] = {
                camera_mapping[old_id]: data
                for old_id, data in projections.items()
                if old_id in camera_mapping
            }
    return input_dict

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, default='individual', help='Output directory for individual annotation files')
    parser.add_argument('--offset_path', type=str, required=True, help='Camera offsets json')

    args = parser.parse_args()
    camera_mapping = {f'cvlabrpi{i}':f'cam{new}' for new, i in enumerate([10,21,13,12,7,19,24,5,23,3,2,4,1,22,11,8,26,17,14,25,6,9,18,15,16])}

    with open(args.input_json, 'r') as f:
        input_dict = json.load(f)

    output_dict = trim_offset(input_dict, args.offset_path)

    output_dict = change_camera_names(output_dict, camera_mapping)

    with open(Path(args.output), 'w') as f:
        json.dump(output_dict, f)


if __name__ == "__main__":
    main()