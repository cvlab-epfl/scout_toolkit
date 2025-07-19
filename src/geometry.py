import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from collections import namedtuple

Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])

def project_world_to_camera(world_point, calib):
    """
    Project a 3D point from world coordinates to image plane (pixel coordinates).
    
    Uses OpenCV's projectPoints function with camera calibration parameters including
    intrinsic matrix, rotation, translation, and distortion coefficients.
    
    Args:
        world_point (array-like): 3D point in world coordinates [X, Y, Z].
                                 Can be a list, tuple, or numpy array.
        calib (object): Camera calibration object containing:
                       - K: 3x3 intrinsic camera matrix
                       - R: 3x3 rotation matrix (world to camera)
                       - T: 3x1 translation vector (world to camera)
                       - dist: distortion coefficients
    
    Returns:
        numpy.ndarray: 2D point in image coordinates [u, v] as a flattened array.
    
    Example:
        >>> world_pt = [1.0, 2.0, 5.0]  # 3D point 5 meters away
        >>> pixel_pt = project_world_to_camera(world_pt, camera_calib)
        >>> print(f"Pixel coordinates: {pixel_pt}")
    """
    # Ensure world_point is a proper 3x1 float32 column vector
    world_point = np.array(world_point).reshape(3, 1).astype(np.float32)
    
    # Extract and format calibration parameters
    K1 = np.array(calib.K).reshape(3, 3).astype(np.float32)
    R1 = R.from_matrix(calib.R).as_rotvec().astype(np.float32)  # Convert to rotation vector
    T1 = np.array(calib.T).reshape(3, 1).astype(np.float32)
    D1 = np.array(calib.dist).reshape(-1, 1).astype(np.float32)
    
    # Project 3D point to 2D using OpenCV
    point1, _ = cv2.projectPoints(world_point, R1, T1, K1, D1)
    
    return point1.squeeze()


def get_ray_directions(points_2d, multi_calib):
    """
    Compute 3D ray origins and directions from 2D image points for multiple cameras.
    
    For each 2D point and corresponding camera, this function:
    1. Undistorts the 2D point using camera calibration
    2. Converts to homogeneous coordinates
    3. Computes the ray origin (camera center in world coordinates)
    4. Computes the ray direction in world coordinates
    
    Args:
        points_2d (list): List of 2D points, one per camera. Each point should be
                         array-like with [u, v] pixel coordinates.
        multi_calib (list): List of camera calibration objects, one per camera.
                           Each calibration object should contain K, R, T, and dist.
    
    Returns:
        tuple: (ray_origins, ray_directions)
            - ray_origins (list): List of 3D ray origin points (camera centers)
            - ray_directions (list): List of 3D ray direction vectors
    
    Note:
        The function assumes the same number of 2D points and calibration objects.
        Ray directions are not normalized - use compute_raymesh_intersection for
        normalized rays.
    
    Example:
        >>> points_2d = [[320, 240], [315, 235]]  # 2D points from 2 cameras
        >>> origins, directions = get_ray_directions(points_2d, camera_calibs)
    """
    undistorted_points = []
    
    # Undistort each 2D point using its corresponding camera calibration
    for point_2d, calib in zip(points_2d, multi_calib):
        undistorted = cv2.undistortPoints(
            np.array(point_2d, dtype=np.float32), 
            calib.K, 
            calib.dist, 
            P=calib.K
        )
        undistorted_points.append(undistorted.squeeze())
    
    # Convert to homogeneous coordinates (add z=1)
    homogenous = np.hstack([
        np.array(undistorted_points), 
        np.ones((len(undistorted_points), 1))
    ])
    
    # Compute ray origins: camera centers in world coordinates
    # Camera center = -R^T * T (inverse of extrinsic transformation)
    ray_origin = [-calib.R.T @ calib.T for calib in multi_calib]
    
    # Compute ray directions in world coordinates
    # Direction = R^T * K^(-1) * homogeneous_point
    ray_direction = [
        calib.R.T @ np.linalg.inv(calib.K) @ point.T 
        for point, calib in zip(homogenous, multi_calib)
    ]
    
    return ray_origin, ray_direction


def compute_raymesh_intersection(ray_origin, ray_direction, mesh):
    """
    Compute intersection points between a 3D ray and a mesh.
    
    Uses the mesh's built-in ray intersection functionality to find all points
    where the ray intersects the mesh surface.
    
    Args:
        ray_origin (array-like): 3D starting point of the ray [X, Y, Z].
        ray_direction (array-like): 3D direction vector of the ray [dX, dY, dZ].
                                   Will be normalized internally.
        mesh (trimesh.Trimesh): 3D mesh object with ray intersection capability.
    
    Returns:
        numpy.ndarray: Array of 3D intersection points. Shape: (N, 3) where N is
                      the number of intersections found.
    
    Note:
        The ray direction is normalized before intersection computation.
        Multiple intersections may be returned if the ray passes through
        the mesh multiple times.
    
    Example:
        >>> import trimesh
        >>> mesh = trimesh.load('model.obj')
        >>> intersections = compute_raymesh_intersection([0, 0, 0], [1, 0, 0], mesh)
    """
    # Normalize the ray direction vector
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    # Perform ray-mesh intersection using trimesh
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin],
        ray_directions=[ray_direction]
    )
    
    return locations


def filter_and_sort_elements(elements, distances, min_dist=None, max_dist=None):
    """
    Filter and sort elements based on their distance to the camera.
    
    Filters out elements outside the specified distance bounds and sorts
    the remaining elements by distance in ascending order (closest first).
    
    Args:
        elements (list): List of elements to filter and sort.
        distances (list): List of distances corresponding to each element.
                         Must have the same length as elements.
        min_dist (float, optional): Minimum distance threshold (inclusive).
                                   Elements closer than this are filtered out.
                                   If None, no minimum filter is applied.
        max_dist (float, optional): Maximum distance threshold (inclusive).
                                   Elements farther than this are filtered out.
                                   If None, no maximum filter is applied.
    
    Returns:
        list: Filtered and sorted list of elements, ordered by distance
              from closest to furthest.
    
    Example:
        >>> points = ['A', 'B', 'C', 'D']
        >>> dists = [5.0, 2.0, 8.0, 3.0]
        >>> filtered = filter_and_sort_elements(points, dists, min_dist=2.5, max_dist=7.0)
        >>> # Returns ['D', 'A'] (distances 3.0, 5.0)
    """
    # Filter elements based on distance bounds
    filtered_elements = [
        (elem, dist) for elem, dist in zip(elements, distances) 
        if (min_dist is None or dist >= min_dist) and 
           (max_dist is None or dist <= max_dist)
    ]
    
    # Sort by distance in ascending order (closest first)
    sorted_elements = sorted(filtered_elements, key=lambda x: x[1])
    
    # Return only the elements, discarding the distances
    return [elem for elem, _ in sorted_elements]


def project_2d_points_to_mesh(points_2d, calibs, mesh):
    """
    Project 2D image points to 3D mesh surface using ray-mesh intersection.
    
    For each 2D point and camera pair:
    1. Computes the 3D ray from camera through the 2D point
    2. Finds intersections with the mesh
    3. Filters intersections by depth and height constraints
    4. Returns the closest valid intersection point
    
    Args:
        points_2d (list): List of 2D points in image coordinates.
                         Each point should be array-like [u, v].
        calibs (list): List of camera calibration objects corresponding to points_2d.
        mesh (trimesh.Trimesh): 3D mesh to intersect rays with.
    
    Returns:
        list: List of 3D world points on the mesh surface. Each element is either:
              - A 3D point [X, Y, Z] if intersection found and valid
              - None if no valid intersection found
    
    Note:
        The function applies several filtering criteria:
        - Minimum depth of 1 unit from camera
        - Height-based filtering (points above 0.8 units are filtered to below 0.8)
        - Returns the closest valid intersection point
    
    Example:
        >>> points_2d = [[320, 240], [315, 235]]
        >>> ground_points = project_2d_points_to_mesh(points_2d, calibs, mesh)
        >>> print(f"Found {len([p for p in ground_points if p is not None])} valid points")
    """
    # Get ray origins and directions for all 2D points
    ray_origins, ray_directions = get_ray_directions(points_2d, calibs)
    
    ground_points = []
    
    for i, (ray_origin, ray_direction, calib) in enumerate(zip(ray_origins, ray_directions, calibs)):
        # Find all intersection points with the mesh
        world_points = compute_raymesh_intersection(ray_origin.squeeze(), ray_direction, mesh)
        
        # Compute depth for each intersection point
        # Depth = z-component of point in camera coordinates
        depths = [
            (-((-np.array(calib.R) @ np.array(inter_point).reshape(3, 1)) - 
               np.array(calib.T).reshape(3, 1)))[2][0]  
            for inter_point in world_points
        ]
        
        # Filter points by minimum depth (remove points too close to camera)
        world_points_filtered = filter_and_sort_elements(world_points, depths, min_dist=1)
        
        # Apply height-based filtering: if points exist above z=0.8, filter to below z=0.8
        if len(world_points_filtered) > 0 and world_points_filtered[0][2] > 0.8:
            world_points_filtered = [
                point for point in world_points_filtered if point[2] < 0.8
            ]
        
        # Select the closest valid point or None if no valid points
        if len(world_points_filtered) == 0:
            ground_points.append(None)
        else:        
            ground_points.append(world_points_filtered[0])
    
    return ground_points


def bbox_to_mesh(bbox, calibration, mesh):
    """
    Convert a 2D bounding box to a 3D point on the mesh surface.
    
    Takes the bottom-center point of the bounding box (typically representing
    the ground contact point of an object) and projects it to the mesh surface.
    
    Args:
        bbox (dict or tuple): Bounding box specification. Can be:
                             - dict with keys 'x', 'y', 'w', 'h'
                             - tuple/list with (x, y, w, h)
                             Where (x,y) is top-left corner, (w,h) is width/height
        calibration (object): Camera calibration object for the image containing the bbox.
        mesh (trimesh.Trimesh): 3D mesh to project onto.
    
    Returns:
        list: Single-element list containing either:
              - 3D point [X, Y, Z] on mesh surface if projection successful
              - None if projection failed
    
    Note:
        The function uses the bottom-center point of the bounding box
        (x + w/2, y + h) as the 2D point for projection.
    
    Example:
        >>> bbox = {'x': 100, 'y': 50, 'w': 80, 'h': 120}
        >>> result = bbox_to_mesh(bbox, camera_calib, mesh)
        >>> if result[0] is not None:
        ...     print(f"Object ground point: {result[0]}")
    """
    # Handle both dictionary and tuple/list formats
    if isinstance(bbox, dict):
        bbox = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    
    x, y, w, h = bbox
    
    # Use bottom-center point of bounding box (typical ground contact point)
    points_2d = (x + w/2, y + h)
    
    # Project to mesh and return result
    return project_2d_points_to_mesh([points_2d], [calibration], mesh)