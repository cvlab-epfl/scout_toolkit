from .loading import get_annotations_for_frame_and_camera, load_mot, load_coco, load_individual
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

def convert_2d_bbox_to_3d_cylinder(detection_data, calib, mesh):
    """
    Convert a 2D bounding box into a 3D cylinder in world coordinates.
    
    detection_data: dict, contains 'bboxes', 'labels', 'scores', 'instances_id'
    calibs: list of Calibration objects (one per camera)
    mesh: 3D mesh of the scene for ray-mesh intersection.
    
    Returns a list of 3D cylinders (each represented as (base_center, radius, height)).
    """
    cylinders = []

    # plt.figure(figsize=(15,15))
    # plt.imshow(frame)
    for i, bbox in enumerate(detection_data['bboxes']):
        # Get the bounding box
        x_min, y_min, x_max, y_max = bbox

        # Midpoint of the bottom of the bounding box (ground level)
        bottom_center_2d = np.array([[(x_min + x_max) / 2, y_max]])
        # plt.scatter(bottom_center_2d[:,0], bottom_center_2d[:,1])
        # Project the bottom center of the bounding box to 3D space (ground level)
        bottom_center_3d = project_2d_points_to_mesh(bottom_center_2d, [calib], mesh, show_intersections=False)[0]

        # If no intersection is found, skip this detection
        if bottom_center_3d is None:
            print(f"Skipping detection {i}, no ground intersection found.")
            cylinders.append(None)
            continue

        # Get the calibration for the current camera
        # calib = calibs[i]
        R = np.array(calib.R)
        T = np.array(calib.T).reshape(3, 1)

        # Convert bottom_center_3d from world to camera coordinates
        # P_camera = R * (P_world - T)
        bottom_center_3d_homogeneous = np.array(bottom_center_3d).reshape(3, 1)  # 3x1 vector
        bottom_center_cam = -((-R@bottom_center_3d_homogeneous) - T)

        # print(bottom_center_3d_homogeneous, bottom_center_cam)
        
        # Depth is the Z-coordinate in the camera coordinate system
        depth = bottom_center_cam[2][0]  # Extract the Z value

        # Radius in 2D (half the width of the bounding box in image plane)
        bbox_width_2d = np.abs(x_max - x_min)
        radius_2d = bbox_width_2d / 2

        # Estimate the radius in world coordinates using the depth of the bottom_center_cam
        fx = calib.K[0][0]  # Focal length in x direction (from intrinsic matrix)
        radius_3d = (radius_2d / fx) * depth

        # Height in 2D (vertical height of the bounding box in image plane)
        bbox_height_2d = np.abs(y_max - y_min)

        # Estimate the height in world coordinates using the same depth factor
        height_3d = (bbox_height_2d / fx) * depth

        # print(depth, radius_2d, radius_3d, bbox_height_2d, height_3d,)
        # Store the cylinder (base center, radius, height)
        cylinders.append((bottom_center_3d, radius_3d, height_3d))

    return cylinders

def visualize_cylinders_with_labels(mesh, cylinders, detection_data):
    """
    Visualize the 3D mesh and the cylinders, each colored by instance ID, with text labels.
    
    mesh: The 3D mesh of the scene.
    cylinders: List of tuples (base_center, radius, height), where each represents a cylinder.
    detection_data: Contains 'labels', 'scores', 'instances_id' for text annotations.
    """
    # Prepare mesh for plotting: extract vertices and faces
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    i, j, k = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]
    
    # Create 3D plot with Plotly
    fig = go.Figure()

    # Add mesh surface
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        opacity=0.5,
        color='lightblue',
        name='Mesh'
    ))

    # Add cylinders and annotations
    for idx, (cylinder, instance_id) in enumerate(zip(cylinders, detection_data['instances_id'])):
        base_center, radius, height = cylinder
        label = detection_data['labels'][idx]
        score = detection_data['scores'][idx]
        color = "rgb({}, {}, {})".format(*get_color_for_instance(instance_id))


        # print(instance_id, color, get_color_for_instance(instance_id))
    #         # Generate a unique color for each instance id
    # def get_color_for_instance(instance_id):
    #     np.random.seed(instance_id)  # Set seed based on instance_id to ensure consistency
    #     return tuple(np.random.randint(0, 256, 3).tolist())  # Random color in BGR format
    
    # def get_color_for_instance(instance_id):
    #     # Function to get a color based on instance ID (simple hash to RGB)
    #     np.random.seed(instance_id)
    #     return "rgb({}, {}, {})".format(*np.random.randint(0, 255, size=3))
    
        # Cylinder top and bottom centers
        base_x, base_y, base_z = base_center
        top_z = base_z + height
        
        # Create circle points for the base and top of the cylinder
        angles = np.linspace(0, 2 * np.pi, 20)
        circle_x = radius * np.cos(angles) + base_x
        circle_y = radius * np.sin(angles) + base_y
        circle_bottom_z = np.full_like(circle_x, base_z)
        circle_top_z = np.full_like(circle_x, top_z)

        # Plot the cylinder's base and top circles
        fig.add_trace(go.Scatter3d(x=circle_x, y=circle_y, z=circle_bottom_z,
                                   mode='lines', line=dict(color=color, width=2),
                                   name=f'Base of Cylinder {instance_id}'))

        fig.add_trace(go.Scatter3d(x=circle_x, y=circle_y, z=circle_top_z,
                                   mode='lines', line=dict(color=color, width=2),
                                   name=f'Top of Cylinder {instance_id}'))

        # Plot vertical lines (edges of the cylinder)
        for j in range(len(circle_x)):
            fig.add_trace(go.Scatter3d(
                x=[circle_x[j], circle_x[j]], y=[circle_y[j], circle_y[j]], z=[base_z, top_z],
                mode='lines', line=dict(color=color, width=2),
                showlegend=False))

        # Add text annotation next to the cylinder base
        annotation_text = f"ID: {instance_id}, Label: {label}, Score: {score:.2f}"
        fig.add_trace(go.Scatter3d(
            x=[base_x], y=[base_y], z=[base_z],
            mode='text', text=[annotation_text],
            textposition="top right",
            showlegend=False))

    # Set plot labels and layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (World)',
            yaxis_title='Y (World)',
            zaxis_title='Z (World, Up)'
        ),
        title="3D Cylinders with Instance ID Annotations",
        autosize=True,
        showlegend=False
    )

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
    # Desired target point coordinates
    target_x, target_y, target_z = 30, 140, 0

    # Define offset relative to the target point in the original coordinate space
    offset_x, offset_y, offset_z = 0, 0, 10  # You can adjust these values as needed

    # Calculate the total range of the scene
    scene_range_x = x_range[1] - x_range[0]
    scene_range_y = y_range[1] - y_range[0]
    scene_range_z = z_range[1] - z_range[0]

    # Normalize the offset relative to the scene extent
    normalized_eye_x = offset_x / scene_range_x
    normalized_eye_y = offset_y / scene_range_y
    normalized_eye_z = offset_z / scene_range_z

    # Set the scene camera to focus on the target point with the desired offset
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode='cube',
        ),
        scene_camera=dict(
            eye=dict(
                x=target_x + normalized_eye_x,
                y=target_y + normalized_eye_y,
                z=target_z + normalized_eye_z
            ),
            center=dict(x=target_x, y=target_y, z=target_z),
            up=dict(x=0, y=0, z=1)
        ),
        title="3D Scene with Camera Targeting Specific Point"
    )

    
    # Update plot layout for equal scaling
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode='cube',  # Ensures equal scaling of all axes
        ),
        scene_camera=dict(
            eye=dict(x=30, y=150, z=0),
            center=dict(x=30, y=150, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        title="Ray-Mesh Intersection"
    )
    
    # Display the plot
    fig.show()
    return fig

def visualize_cylinders_2d_perspective_with_background(cylinders, calib, background_image_path, detection_data):
    """
    Visualize 3D cylinders in 2D with a perspective effect and an image background.
    
    cylinders: List of tuples (base_center, radius, height) representing the cylinders.
    calib: Calibration object containing intrinsic (K), rotation (R), translation (T), and distortion (dist) parameters.
    background_image_path: Path to the background image (e.g., the captured frame).
    detection_data: Contains 'labels', 'scores', 'instances_id' for text annotations and colors.
    """
    # Load the background image
    background_image = cv2.imread(background_image_path)
    if background_image is None:
        raise ValueError(f"Image at {background_image_path} could not be loaded.")
    
    background_image_rgb = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = background_image.shape

    # Set up the plot with the image as background
    fig, ax = plt.subplots()
    ax.imshow(background_image_rgb, extent=[0, img_width, img_height, 0])  # Flip Y axis for image coordinates

    for idx, cylinder in enumerate(cylinders):
        base_center, radius, height = cylinder
        instance_id = detection_data['instances_id'][idx]
        label = detection_data['labels'][idx]
        score = detection_data['scores'][idx]
        color = get_color_for_instance(instance_id)

        # Cylinder base center and top in world coordinates
        base_x, base_y, base_z = base_center
        top_z = base_z + height

        # Generate points around the base and top circles
        angles = np.linspace(0, 2 * np.pi, 20)
        circle_base_x = radius * np.cos(angles) + base_x
        circle_base_y = radius * np.sin(angles) + base_y
        circle_top_x = radius * np.cos(angles) + base_x
        circle_top_y = radius * np.sin(angles) + base_y

        # Convert 3D points to 2D using the camera calibration
        base_2d = [project_world_to_camera([circle_base_x[i], circle_base_y[i], base_z], calib) for i in range(len(angles))]
        top_2d = [project_world_to_camera([circle_top_x[i], circle_top_y[i], top_z], calib) for i in range(len(angles))]

        # Unpack the 2D coordinates
        base_2d_x, base_2d_y = zip(*base_2d)
        top_2d_x, top_2d_y = zip(*top_2d)

        # Plot the base and top circles in 2D with instance-specific color
        ax.plot(base_2d_x, base_2d_y, color=np.array(color) / 255.0, label=f'Base of Cylinder {instance_id}', alpha=0.3)
        ax.plot(top_2d_x, top_2d_y, color=np.array(color) / 255.0, label=f'Top of Cylinder {instance_id}', alpha=0.3)

        # Plot the vertical lines connecting the base and top
        for j in range(len(base_2d_x)):
            ax.plot([base_2d_x[j], top_2d_x[j]], [base_2d_y[j], top_2d_y[j]], color=np.array(color) / 255.0, alpha=0.3)

        # Add text annotation near the cylinder base
        annotation_text = f"ID: {instance_id}, Label: {label}, Score: {score:.2f}"
        ax.text(base_2d_x[0], base_2d_y[0], annotation_text, color=np.array(color) / 255.0, fontsize=9)

    # Set equal scaling and labels
    ax.set_aspect('equal')
    ax.set_xlabel("X (Image Coordinates)")
    ax.set_ylabel("Y (Image Coordinates)")
    ax.set_title("2D Perspective Projection of 3D Cylinders with Background")

    plt.show()

@dataclass
class Trajectory:
    camera:str = None
    id:str = None
    detections:np.ndarray = None
    person_id = None

    def get_detections(self) -> np.ndarray:
        if isinstance(self.detections, list):
            assert (isinstance(self.detections[0], np.ndarray) and self.detections[0].shape[0] == 3) or (isinstance(self.detections[0], list) and len(self.detections[0]) == 3), 'Input trajectory must be n by 3'
            return np.array(self.detections)
        elif isinstance(self.detections, np.ndarray):
             return self.detections
        
    def __len__(self) -> int:
        if isinstance(self.detections, list):
            return len(self.detections)
        if isinstance(self.detections, np.ndarray):
            return self.detections.shape[1]

    def set_person_id(self, id) -> None:
        self.person_id = id

    def get_valid_detections(self) -> np.ndarray:
        detections = self.get_detections()
        valid_indices = np.where(np.any(detections != 0, axis=1))[0]

        valid_detections = detections[valid_indices]

        return valid_detections
    
    def remove_faulty_detections(self, distance_threshold = 5.0) -> None:
        positions_array = self.get_detections()

        # Create a mask for valid positions (not all zeros)
        valid_mask = ~np.all(positions_array == 0, axis=1)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            return None

        # Copy positions to avoid modifying the original data
        positions_processed = positions_array.copy()
        for i in range(positions_array.shape[0] - 1):
            pos1 = positions_array[i, :]
            pos2 = positions_array[i+1, :]
            if np.any(pos2 != 0) and np.any(pos1 != 0):
                dist = np.linalg.norm(pos2 - pos1)
                if dist > distance_threshold:
                    positions_processed[i+1, :] = [0,0,0]
        self.detections = positions_processed
        return None


def preprocess_trajectory(trajectory:Trajectory, distance_threshold = 5.0):
    
    positions_array = trajectory.detections.copy()

    # Create a mask for valid positions (not all zeros)
    valid_mask = ~np.all(positions_array == 0, axis=1)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        return trajectory

    # Copy positions to avoid modifying the original data
    positions_processed = positions_array.copy()
    for i in range(positions_array.shape[0] - 1):
        pos1 = positions_array[i, :]
        pos2 = positions_array[i+1, :]
        if np.any(pos2 != 0) and np.any(pos1 != 0):
            dist = np.linalg.norm(pos2 - pos1)
            if dist > distance_threshold:
                positions_processed[i+1, :] = [0,0,0]

    return positions_processed
    
def visualize_matched_tracks_3d(mesh, trajectories:List[Trajectory]):
    # trajectories = [preprocess_trajectories(trajectory) for trajectory in trajectories]
    # Prepare mesh for plotting: extract vertices and faces
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    i, j, k = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]
    
    # Create 3D plot with Plotly
    fig = go.Figure()

    # Add mesh surface
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        opacity=0.5,
        color='lightblue',
        name='Mesh'
    ))
    # for id1, id2 in matched_ids:
    #     color_rgb = get_color_for_instance(id1)
    #     trajectory1 = trajectories[0][id1]
    #     trajectory2 = trajectories[1][id2]

        
    for trajectory in trajectories:

        person_id = trajectory.person_id
        color_rgb = get_color_for_instance(person_id)

        color = f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
        # Filter out zero positions (assuming zeros represent missing data)
        valid_indices = np.where(np.any(trajectory != 0, axis=1))[0]

        if len(valid_indices) == 0:
            continue  # Skip if no valid data

        valid_positions = trajectory.get_valid_detections()
        x_traj = valid_positions[:, 0]
        y_traj = valid_positions[:, 1]
        z_traj = valid_positions[:, 2]
        
        # Plot the trajectory as a line
        fig.add_trace(go.Scatter3d(
            x=x_traj, y=y_traj, z=z_traj,
            mode='lines',
            line=dict(color=color, width=2),
            name=f'Trajectory {person_id}'
        ))

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
    # Desired target point coordinates
    target_x, target_y, target_z = 30, 140, 0

    # Define offset relative to the target point in the original coordinate space
    offset_x, offset_y, offset_z = 0, 0, 10  # You can adjust these values as needed

    # Calculate the total range of the scene
    scene_range_x = x_range[1] - x_range[0]
    scene_range_y = y_range[1] - y_range[0]
    scene_range_z = z_range[1] - z_range[0]

    # Normalize the offset relative to the scene extent
    normalized_eye_x = offset_x / scene_range_x
    normalized_eye_y = offset_y / scene_range_y
    normalized_eye_z = offset_z / scene_range_z

    # Set the scene camera to focus on the target point with the desired offset
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode='cube',
        ),
        # scene_camera=dict(
        #     eye=dict(
        #         x=target_x + normalized_eye_x,
        #         y=target_y + normalized_eye_y,
        #         z=target_z + normalized_eye_z
        #     ),
        #     center=dict(x=target_x, y=target_y, z=target_z),
        #     up=dict(x=0, y=0, z=1)
        # ),
        title="3D Scene with Camera Targeting Specific Point"
    )
    print('Camera: ', fig.layout.scene.camera)
    fig.show()