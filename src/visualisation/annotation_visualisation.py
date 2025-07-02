# from .config import IMAGEDIR, ROOTDIR
from typing import Optional, Union
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go

def get_color_for_instance(instance_id):
    # Generate a unique color for each instance id
    np.random.seed(instance_id)  # Set seed based on instance_id to ensure consistency
    return tuple(np.random.randint(0, 256, 3).tolist())  # Random color in BGR format

def draw_annotation(image, bbox, track_id, thickness = 2):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    x, y, w, h = bbox
    top_left = (int(x), int(y))
    bottom_right = (int(x + w), int(y + h))

    # Get color based on the instance ID
    color = get_color_for_instance(track_id)[::-1]
    # Draw the bounding box
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    label = f"ID: {track_id}"
    
    # Position for text
    text_position = (int(x), int(y) - 10)

    # Put the label and score above the bounding box
    cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

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

        # Sorting by depth and filtering
        depths = [(-((-np.array(calib.R)@np.array(inter_point).reshape(3, 1)) - np.array(calib.T).reshape(3, 1)))[2][0]  for inter_point in world_points]
        world_points_filtered = filter_and_sort_elements(world_points, depths, min_dist=1)
        # hacky remove point with high z

        if len(world_points_filtered) > 0 and world_points_filtered[0][2] > 0.8:
            print(f"Warning the intersection with height (z-component) higher than 0.8 have been removed for points_2d - {i}: {points_2d[i]}")
            world_points_filtered = [point for point in world_points_filtered if point[2] < 0.8]
                
        if len(world_points_filtered) == 0:
            print(f"No intersection found for point {i}, filling with None")
            ground_points.append(None)
        else:        
            ground_points.append(world_points_filtered[0])

    return ground_points

def bbox_to_mesh(bbox, calibration, mesh):
    if isinstance(bbox, dict):
                bbox = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    x, y, w, h = bbox
    points_2d = (x + w/2, y + h)
    return project_2d_points_to_mesh([points_2d], [calibration], mesh)

def project_world_to_camera(world_point, calib, unnormalized=False, width=None, height=None):
    """
    Project 3D point world coordinate to image plane (pixel coordinate)
    """
    K1, R1, T1, D1 = calib.K, calib.R, calib.T, calib.dist

    world_point = np.array(world_point).reshape(3,1).astype(np.float32)
    K1 = np.array(K1).reshape(3,3).astype(np.float32)
    R1 = R.from_matrix(R1).as_rotvec().astype(np.float32)
    # R1 = np.array(R1).reshape(3,1).astype(np.float32)
    T1 = np.array(T1).reshape(3,1).astype(np.float32)
    
    D1 = np.array(D1).reshape(-1,1).astype(np.float32)
        
    point1, _ = cv2.projectPoints(world_point, R1, T1, K1, D1)

    return point1.squeeze()


def draw_3d(points3d, mesh):
    # Convert mesh to vertices and faces
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Create mesh3d object
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.5,
        name='Mesh'
    )

    # Create scatter3d object for the points
    scatter = go.Scatter3d(
        x=points3d[:, 0],
        y=points3d[:, 1],
        z=points3d[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Points'
    )

    fig = go.Figure(data=[mesh_plot, scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
        ),
        width=800,
        height=700,
        title='3D Mesh and Points'
    )
    fig.show()
