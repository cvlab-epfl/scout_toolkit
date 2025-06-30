import json
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List

class AnnotationDataset(Dataset):
    """
    Usage:
    dataset = AnnotationDataset("annotations.json")

    dataloader = DataLoader(
        dataset,
        batch_size=8,         # or 1 for per-frame iteration
        shuffle=False,        # set to True if needed
        collate_fn=collate_fn
    )

    # Example loop
    for batch in dataloader:
        print(batch["frame_ids"])
        print(batch["annotations"])  # List[List[annotation dicts]]
    """
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        self.sequence_id = data["sequence_id"]
        self.total_frames = data["total_frames"]
        self.frames = data["frames"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame = self.frames[idx]
        frame_id = frame["frame_id"]
        timestamp = frame["timestamp_ms"]
        annotations = frame.get("annotations", [])

        # Parse annotations into a more uniform structure
        parsed_annotations = []
        for ann in annotations:
            entry = {
                "track_id": ann["track_id"],
                "cuboid_3d": ann["cuboid_3d"],
                "projections_2d": ann["projections_2d"]
            }
            parsed_annotations.append(entry)

        return {
            "frame_id": frame_id,
            "timestamp_ms": timestamp,
            "annotations": parsed_annotations
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    frame_ids = [item["frame_id"] for item in batch]
    timestamps = [item["timestamp_ms"] for item in batch]
    all_annotations = [item["annotations"] for item in batch]

    return {
        "frame_ids": frame_ids,
        "timestamps": timestamps,
        "annotations": all_annotations
    }

