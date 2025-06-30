from collections import namedtuple
import json
import .config as config
from typing import List
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2




    

    


class Camera:
    def __init__(self, camera_name):
        self.camera_name = camera_name
        self.calib_path = Path(config.CALIBPATH) / f"{camera_name}.json"

   
        

    def project_from(self, frame_annotations: Union[Dict, List[Dict]], mesh:trimesh.Mesh) -> List[Dict]:
        """
        Projects 2D bounding boxes into 3D
        """
        if isinstance(frame_annotations, dict):
            frame_annotations = [frame_annotations]

        results = []
        calib = self.calibration
        bases = [((x2 - x1) / 2, y2) for annotation in frame_annotations for x1, y1, x2, y2 in [annotation['bbox']]]

        world_coords = project_2d_points_to_mesh(bases, self.calibration, mesh)

        results = [{"track_id":annotation["track_id"], "world":world_coords[i]} for i, annotation in enumerate(frame_annotations)]

        return results

    def show_frame(self, frame_id, annotations_3d: Optional[Union[Dict, List[Dict]]] = None, annotations_2d: Optional[Union[Dict, List[Dict]]] = None) -> None:
        """
        Display the frame with 3D and 2D annotations drawn over it.
        """
        # Load image
        image_path = Path(config.IMAGEPATH) / self.camera_name / f"image_{frame_id}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Frame image not found at {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_2d = image_rgb.copy()
        # Handle 3D annotations
        if annotations_3d:
            if annotations_2d:
                if isinstance(annotations_2d, dict):
                    annotations_2d = [annotations_2d]
                else:
                    annotations_2d = []
            if isinstance(annotations_3d, dict):
                annotations_3d = [annotations_3d]

            annotations_2d.append(project_to(annotations_3d))

        # Handle 2D annotations
        if annotations_2d:
            if isinstance(annotations_2d, dict):
                annotations_2d = [annotations_2d]

            
            for ann in annotations_2d:
                x1, y1, x2, y2 = map(int, ann["bbox"])
                color = get_color_for_instance(ann["track_id"])
                cv2.rectangle(image_2d, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_2d, f"ID: {ann['track_id']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Show image with 2D boxes
        plt.figure(figsize=(10, 8))
        plt.imshow(image_2d)
        plt.title(f"{self.camera_name} Frame {frame_id} - 2D Annotations")
        plt.axis("off")
        plt.show()
