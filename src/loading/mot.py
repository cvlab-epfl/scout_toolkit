from pathlib import Path

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