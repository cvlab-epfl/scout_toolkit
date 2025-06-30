from collections import namedtuple
import json
import .config as config
from typing import List
from copy import deepcopy
Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])

def project_world_to_camera(world_point, calib):
    """
    Project 3D point world coordinate to image plane (pixel coordinate)
    """
    world_point = np.array(world_point).reshape(3,1).astype(np.float32)
    K1 = np.array(calib.K).reshape(3,3).astype(np.float32)
    R1 = R.from_matrix(calib.R).as_rotvec().astype(np.float32)
    T1 = np.array(calib.T).reshape(3,1).astype(np.float32)
    
    D1 = np.array(calib.dist).reshape(-1,1).astype(np.float32)
        
    point1, _ = cv2.projectPoints(world_point, R1, T1, K1, D1)

    return point1.squeeze()

def get_ray_directions(points_2d, multi_calib):

    undistorted_points = []
    for point_2d, calib in zip(points_2d, multi_calib):
        undistorted = cv2.undistortPoints(np.array(point_2d, dtype=np.float32), calib.K, calib.dist, P=calib.K)
        undistorted_points.append(undistorted.squeeze())

    homogenous = homogenous = np.hstack([np.array(undistorted_points), np.ones((len(undistorted_points), 1))])
 
    ray_origin = [-calib.R.T @ calib.T for calib in multi_calib]

    ray_direction = [calib.R.T @ np.linalg.inv(calib.K) @ point.T for point, calib in zip(homogenous, multi_calib)]

    return ray_origin, ray_direction

def compute_raymesh_intersection(ray_origin, ray_direction, mesh):
    # Normalize the ray direction
    ray_direction /= np.linalg.norm(ray_direction)
    
    # Perform ray-mesh intersection
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin],
        ray_directions=[ray_direction]
    )

    return locations
    
def filter_and_sort_elements(elements, distances, min_dist=None, max_dist=None):
    """
    Filters and sorts elements based on their distance to the camera.

    Parameters:
    elements (list): List of elements.
    distances (list): List of distances corresponding to the elements.
    min_dist (float or None): Minimum distance threshold (inclusive). If None, no minimum filter is applied.
    max_dist (float or None): Maximum distance threshold (inclusive). If None, no maximum filter is applied.

    Returns:
    list: Filtered and sorted list of elements by distance (closest to furthest).
    """
    
    # Filter out elements outside the distance bounds
    filtered_elements = [
        (elem, dist) for elem, dist in zip(elements, distances) 
        if (min_dist is None or dist >= min_dist) and (max_dist is None or dist <= max_dist)
    ]

    # Sort elements by distance (ascending)
    sorted_elements = sorted(filtered_elements, key=lambda x: x[1])

    # Return only the elements, without the distances
    return [elem for elem, _ in sorted_elements]
    
def project_2d_points_to_mesh(points_2d, calibs, mesh):
    ray_origins, ray_directions = get_ray_directions(points_2d, calibs)

    ground_points = []
    for i, (ray_origin, ray_direction, calib) in enumerate(zip(ray_origins, ray_directions, calibs)):

        world_points = compute_raymesh_intersection(ray_origin.squeeze(), ray_direction, mesh)
        depths = [(-((-np.array(calib.R)@np.array(inter_point).reshape(3, 1)) - np.array(calib.T).reshape(3, 1)))[2][0]  for inter_point in world_points]
        world_points_filtered = filter_and_sort_elements(world_points, depths, min_dist=1)

        if len(world_points_filtered) > 0 and world_points_filtered[0][2] > 0.8:
            world_points_filtered = [point for point in world_points_filtered if point[2] < 0.8]
                
        if len(world_points_filtered) == 0:
            ground_points.append(None)
        else:        
            ground_points.append(world_points_filtered[0])

    return ground_points

class Camera:
    def __init__(self, camera_name):
        self.camera_name = camera_name
        self.calib_path = Path(config.CALIBPATH) / f"{camera_name}.json"

    @property
    def calibration(self):
        if not hasattr(self, '_calib'):
            with open(self.calib_path, 'r') as f:
                calib_dict = json.load(f)
            K = np.array(calib_dict.get("K")) if calib_dict.get("K") is not None else None
            R = np.array(calib_dict.get("R")) if calib_dict.get("R") is not None else None
            T = np.array(calib_dict.get("T")) if calib_dict.get("T") is not None else None
            dist = np.array(calib_dict.get("dist")) if calib_dict.get("dist") is not None else None
            view_id = self.camera_name

            self._calib = Calibration(K, R, T, dist, view_id)
        return self._calib
        
    def project_to(self, frame_annotations: Union[Dict, List[Dict]]) -> List[Dict]:
        """
        Projects 3D cylindrical annotations into 2D image space as 2D bounding boxes.
        """
        if isinstance(frame_annotations, dict):
            frame_annotations = [frame_annotations]

        results = []
        calib = self.calibration

        for annotation in frame_annotations:
            center = np.array(annotation['world'])  # (X, Y, Z)
            h = annotation['height']
            r = annotation['radius']

            # Generate 8 extreme points on the cylinder (top and bottom, 4 points each)
            angles = np.linspace(0, 2 * np.pi, 5)[:-1]  # 0, 90, 180, 270 degrees
            dx = r * np.cos(angles)
            dy = r * np.sin(angles)

            # Bottom and top Z coordinates
            bottom_z = center[2] - h / 2
            top_z = center[2] + h / 2

            # Points: 4 around bottom, 4 around top
            world_points = []
            for xi, yi in zip(dx, dy):
                world_points.append([center[0] + xi, center[1] + yi, bottom_z])
                world_points.append([center[0] + xi, center[1] + yi, top_z])
            world_points = np.array(world_points)

            # Project all points
            img_pts = []
            for pt in world_points:
                img_pt = project_world_to_camera(pt, calib)
                img_pts.append(img_pt)
            img_pts = np.array(img_pts)

            # Compute bounding box
            x1, y1 = img_pts.min(axis=0)
            x2, y2 = img_pts.max(axis=0)

            results.append({
                "track_id": annotation["track_id"],
                "bbox": (float(x1), float(y1), float(x2), float(y2))
            })

        return results

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
