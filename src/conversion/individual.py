from argparse import ArgumentParser
from pathlib import Path
import json

def convert_bbox(bbox_in):
    (x1, y1), (x2, y2) = bbox_in
    width = x2 - x1
    height = y2 - y1
    return (x1, y1, width, height)

def as_individual(input_dict, output_path) -> None:
    """
    Outputs annotation file per image
    output_path:
        - cam_dir:
            - image{frame_id}.txt

    Format per line:
    <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, <x>, <y>, <z>
    """
    cam_set = set()

    for frame in input_dict['frames']:
        for annotation in frame['annotations']:
            cam_set.update(annotation.get("projections_2d", {}).keys())

    cam_list = sorted(cam_set)    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cam in cam_list:
        cam_dir = output_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        for frame in input_dict['frames']:
            frame_id = frame["frame_id"]
            with open(cam_dir / f"image_{frame_id}.txt", 'w') as f:
                for annotation in frame["annotations"]:
                    if cam not in annotation["projections_2d"]:
                        continue
                    bbox = convert_bbox(annotation["projections_2d"][cam])  # (x, y, w, h)
                    x, y, w, h = bbox
                    world = annotation["cuboid_3d"]
                    line = (
                        f"{annotation['track_id']}, {x:.2f}, {y:.2f}, "
                        f"{w:.2f}, {h:.2f}, -1, {world['Xw']:.2f}, {world['Yw']:.2f}, {world['Zw']:.2f}\n"
                    )
                    f.write(line)

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, default='individual', help='Output directory for individual annotation files')

    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        input_dict = json.load(f)

    as_individual(input_dict, args.output)

if __name__ == "__main__":
    main()