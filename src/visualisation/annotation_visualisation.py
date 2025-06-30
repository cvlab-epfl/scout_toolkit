from .annotation_loading import get_annotations_for_frame_and_camera, load_mot, load_coco, load_individual
from .config import IMAGEDIR, ROOTDIR
from typing import Optional
import numpy as np
import cv2


def get_color_for_instance(instance_id):
    # Generate a unique color for each instance id
    np.random.seed(instance_id)  # Set seed based on instance_id to ensure consistency
    return tuple(np.random.randint(0, 256, 3).tolist())  # Random color in BGR format

def visualize_detections(image_path, detection_data, return_image=False):
    # Function to visualize bounding boxes, labels, and scores on an image
    # Load the image
    image = cv2.imread(image_path)

    # Check if image loaded correctly
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Get the detection details
    bboxes = detection_data['bboxes']
    labels = detection_data['labels']
    scores = detection_data['scores']
    instance_ids = detection_data['instances_id']

    # Loop over each detection
    for i in range(len(bboxes)):
        # Get the bounding box coordinates
        x_min, y_min, x_max, y_max = bboxes[i]

        # Get color based on the instance ID
        color = get_color_for_instance(instance_ids[i])[::-1]

        # Draw the bounding box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

        # Prepare the label and score text
        label = f"ID: {instance_ids[i]} Label: {labels[i]} Score: {scores[i]:.2f}"
        
        # Position for text
        text_position = (int(x_min), int(y_min) - 10)

        # Put the label and score above the bounding box
        cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # If return_image is True, return the resulting image
    if return_image:
        return image
    
    # Convert BGR image to RGB for Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes
    plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

  def load_frame(cam_name, frame_id, root_dir:Union[str, Path]) -> np.ndarray:
    """
    Load image and draw 2D bounding boxes from annotations.

    Args:
        imagepath (str or Path): Path to the image file.
        annotations (list): List of dicts with keys 'bbox' (dict with x, y, w, h) or 
                            'bbox' as a list/tuple of [x, y, w, h].
        color (tuple): Color of the bounding box in BGR.
        thickness (int): Line thickness for the bounding box.

    Returns:
        image (np.ndarray): Annotated image.
    """

    imagepath = root_dir / 'images' / cam_name / f"image_{frame_id}.jpg"

    # Load the image
    image = cv2.imread(str(imagepath))
    if image is None:
        raise FileNotFoundError(f"Could not load image at {imagepath}")

    return image
  

def visualiser_2d(imagepath, annotations, color=(0, 255, 0), thickness=2):
    """
    Load image and draw 2D bounding boxes from annotations.

    Args:
        imagepath (str or Path): Path to the image file.
        annotations (list): List of dicts with keys 'bbox' (dict with x, y, w, h) or 
                            'bbox' as a list/tuple of [x, y, w, h].
        color (tuple): Color of the bounding box in BGR.
        thickness (int): Line thickness for the bounding box.

    Returns:
        image (np.ndarray): Annotated image.
    """
    # Load the image
    image = cv2.imread(str(imagepath))
    if image is None:
        raise FileNotFoundError(f"Could not load image at {imagepath}")

    # Draw bounding boxes
    for ann in annotations:
        bbox = ann['bbox']
        if isinstance(bbox, dict):
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        else:
            x, y, w, h = bbox
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    return image

def visualiser_3d_points(annotations, image_size=(800, 800), scale=10, point_radius=3, color=(0, 0, 255)):
    """
    Visualize 3D coordinates (Xw, Yw) on a 2D image plane (Zw assumed 0).

    Args:
        annotations (list): List of dicts with key 'world' = [Xw, Yw, Zw].
        image_size (tuple): Size of the output image (width, height).
        scale (float): Scaling factor to map world coordinates to pixels.
        point_radius (int): Radius of the drawn points.
        color (tuple): BGR color for the points.

    Returns:
        np.ndarray: Annotated 2D image showing 3D world points.
    """
    width, height = image_size
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

    for ann in annotations:
        Xw, Yw, _ = ann['world']
        # Convert world coordinates to image coordinates (centered origin)
        x = int(width // 2 + Xw * scale)
        y = int(height // 2 - Yw * scale)  # Flip Y axis to have +Y upwards
        cv2.circle(image, (x, y), point_radius, color, -1)

    return image

def visualise_coco(filepath, cam_name, frame_id, imagedir:Optional[str|Path] = IMAGEDIR):
    data_dict = load_coco(filepath)
    annotations = get_annotations_for_frame_and_camera(data_dict, frame_id, cam_name)
    framepath = Path(imagedir) / cam_name / f"image_{frame_id}.jpg"
    image = visualiser_2d(framepath, annotations)
    return image

def visualise_mot(filepath, cam_name, frame_id, imagedir:Optional[str|Path] = IMAGEDIR):
    annotations = load_mot(filepath, cam_name)
    annotations = [ann for ann in annotations if ann["frame_id"] == frame_id]
    framepath = Path(imagedir) / cam_name / f"image_{frame_id}.jpg"
    image = visualiser_2d(framepath, annotations)
    return image

def visualise_individual(filepath, cam_name, frame_id, imagedir:Optional[str|Path] = IMAGEDIR):
    annotations = load_individual(filepath, cam_name, frame_id)
    framepath = Path(imagedir) / cam_name / f"image_{frame_id}.jpg"
    image = visualiser_2d(framepath, annotations)
    return image


def visualize_frame_pred(seq, cam, frame_index, return_image=False):
    pred_store = HDF5FrameStore(hdf5_template.format(frame_seq=seq, camera=cam))
    frames_name = pred_store.get_all_frame_names()

    frame_name = frames_name[frame_index]
    frame_pred = pred_store.load_frame(frames_name[frame_index])

    frame_path = image_folder_template.format(frame_seq=seq, camera=cam, filename=frame_name)

    if return_image:
        img_viz = visualize_detections(frame_path, frame_pred, return_image)
        return img_viz
    else:
        visualize_detections(frame_path, frame_pred)
    


def load_frame_from_video(video_path, frame_id):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")
    
    # Set the video position to the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    # Check if the frame was read successfully
    if not ret:
        raise ValueError(f"Frame {frame_id} could not be read from {video_path}")
    
    return frame

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
    
def visualize_intersection(mesh, ray_origin=None, ray_direction=None, intersection_locations=[]):
    # Prepare mesh for plotting: extract vertices and faces
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    i, j, k = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]
    
    # Create 3D plot with plotly
    fig = go.Figure()
    
    # Add mesh surface
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        opacity=0.7,
        color='lightblue',
        name='Mesh'
    ))
    
    # Add ray as a line
    if ray_origin is not None and ray_direction is not None:
        ray_end = ray_origin + ray_direction * 20  # Extend ray for visualization
        fig.add_trace(go.Scatter3d(
            x=[ray_origin[0], ray_end[0]],
            y=[ray_origin[1], ray_end[1]],
            z=[ray_origin[2], ray_end[2]],
            mode='lines',
            line=dict(color='red', width=5),
            name='Ray'
        ))
        
        # Add marker at the ray's origin
        fig.add_trace(go.Scatter3d(
            x=[ray_origin[0]],
            y=[ray_origin[1]],
            z=[ray_origin[2]],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='circle'),
            name='Ray Origin'
        ))

    # Add intersection points (if any)
    if len(intersection_locations) > 0:
        locations = np.array(intersection_locations)
        fig.add_trace(go.Scatter3d(
            x=locations[:, 0],
            y=locations[:, 1],
            z=locations[:, 2],
            mode='markers',
            marker=dict(size=8, color='green'),
            name='Intersection'
        ))
    
    # Calculate the range for all axes and adjust to make them equal
    x_range = [x.min(), x.max()]
    y_range = [y.min(), y.max()]
    z_range = [z.min(), z.max()]
    
    # Find the largest range among x, y, and z
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
    
    # Expand all ranges to match the largest range
    x_mid = (x_range[1] + x_range[0]) / 2
    y_mid = (y_range[1] + y_range[0]) / 2
    z_mid = (z_range[1] + z_range[0]) / 2
    
    x_range = [x_mid - max_range / 2, x_mid + max_range / 2]
    y_range = [y_mid - max_range / 2, y_mid + max_range / 2]
    z_range = [z_mid - max_range / 2, z_mid + max_range / 2]
    
    # Update plot layout for equal scaling
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode='cube',  # Ensures equal scaling of all axes
        ),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),  # Camera looking along z-axis by default
        ),
        title="Ray-Mesh Intersection"
    )
    
    # Display the plot
    fig.show()

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
    
def project_2d_points_to_mesh(points_2d, calibs, mesh, show_intersections=False):
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

        if show_intersections:
            visualize_intersection(mesh, ray_origin.squeeze(), ray_direction, world_points)

    return ground_points

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
    # print(point1.shape)
    # print(point1)

    # if unnormalized:
    #     # print("Unnormalized")
    #     # print(f"point1: {point1.squeeze(0)}")
    #     point1 = unnormalized_image_coordinates(point1.squeeze(0), width, height)


    return point1.squeeze()
