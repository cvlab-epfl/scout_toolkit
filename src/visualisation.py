import numpy as np
import cv2
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
