from pathlib import Path

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