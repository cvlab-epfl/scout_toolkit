import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def project_world_to_camera(world_point, calib):
    """
    Project 3D point world coordinates to image plane (pixel coordinate)
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