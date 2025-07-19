# Scout Dataset Toolkit

A comprehensive Python toolkit for working with the Scout multi-camera pedestrian tracking dataset. This toolkit provides easy-to-use interfaces for loading annotations in multiple formats, visualizing data, and evaluating tracking methods.

## 📦Download the dataset

The [Scout dataset](https://scout.epfl.ch/) is a large-scale multi-camera pedestrian tracking dataset captured in realistic outdoor environments. It can be downloaded from [https://scout.epfl.ch/download](https://scout.epfl.ch/download).

This toolkit provides:

- **Multi-format data loading** - Support for COCO, Individual, and MOT annotation formats
- **Rich visualization tools** - 2D bounding box visualization and 3D world coordinate plotting
- **Comprehensive evaluation metrics** - Both 3D world coordinate and standard MOT metrics
- **Camera calibration utilities** - 3D geometry operations and camera projections
- **PyTorch integration** - Ready-to-use PyTorch datasets and data loaders

## 🔧 Installation

### Option 1: Conda Environment (Recommended)
```bash
conda env create -f environment.yml
conda activate scout-env
```

Then install trackeval:
```bash
pip install --no-deps git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a
```

### Option 2: pip Installation
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

Check out [`example.ipynb`](example.ipynb) for a comprehensive walkthrough of the toolkit's main features:

```python
from src.loading import Individual
from src.visualisation import draw_annotation

# Load annotations for a sequence
loader = Individual(root_dir=".", sequence=1)
annotations = loader.retrieve(frame_id=81, target_cam_id='cam_2')

# Visualize annotated frame
img = loader.annotate_frame(81, 'cam_2')
loader.show_frame(img)
```

## 📁 Key Files and Modules

### 🔄 Data Loading (`src/loading.py`)
Support for multiple annotation formats:

- **`Coco`** - COCO format annotations with full sequence loading
- **`Individual`** - Frame-by-frame text file annotations with caching
- **`MOT`** - MOT Challenge format for tracking evaluation
- **`AnnotationDataset`** - PyTorch-compatible dataset wrapper

```python
# Load different annotation formats
from src.loading import Coco, Individual, MOT

# COCO format - full sequence
coco = Coco(rootdir, sequence=1)
annotations = coco.retrieve(frame_id=0, target_cam_id='cam_0')

# Individual format - per-frame files
individual = Individual(rootdir, sequence=1)
annotations = individual.retrieve(frame_id=81, target_cam_id='cam_2')

# MOT format - tracking sequences
mot = MOT(rootdir, sequence=1)
annotations = mot.retrieve(frame_id=0, target_cam_id='cam_0')
```

### 📊 Visualization (`src/visualisation.py`)
Comprehensive visualization tools:

- **`draw_annotation()`** - Draw 2D bounding boxes with track IDs
- **`draw_3d()`** - Plot 3D world coordinates with mesh overlay
- **`get_color_for_instance()`** - Consistent color mapping for track IDs

```python
from src.visualisation import draw_annotation, draw_3d
import matplotlib.pyplot as plt

# 2D visualization
image = draw_annotation(image, bbox, track_id)
plt.imshow(image)
plt.show()

# 3D visualization with mesh
draw_3d(world_points, mesh)
```

### 📈 Evaluation Metrics (`src/metric.py`)
Comprehensive evaluation suite:

- **`MOTMetricEvaluator`** - Standard MOT metrics (MOTA, MOTP, IDF1, HOTA)
- **`compute_mot_metric()`** - 3D world coordinate evaluation
- **`scout_to_dataframe()`** - Convert Scout annotations to evaluation format
- **`simulate_predictions()`** - Generate synthetic predictions for testing

```python
from src.metric import MOTMetricEvaluator, compute_mot_metric

# MOT evaluation
evaluator = MOTMetricEvaluator()
metrics = evaluator.compute_metrics("SCOUT")

# 3D world coordinate evaluation  
gt_df = scout_to_dataframe(gt_sequence, use_world_coords=True)
pred_df = scout_to_dataframe(pred_sequence, use_world_coords=True)
metrics = compute_mot_metric(gt_df, pred_df, threshold=1.0, nb_gt=len(gt_df))
```

### 🔢 Geometry Operations (`src/geometry.py`)
3D geometry and camera calibration utilities:

- **`project_world_to_camera()`** - Project 3D world points to 2D image coordinates
- **`project_2d_points_to_mesh()`** - Project 2D points to 3D mesh surface
- **`bbox_to_mesh()`** - Convert 2D bounding boxes to 3D world coordinates
- **`load_calibration()`** - Load camera calibration parameters

```python
from src.geometry import project_world_to_camera, bbox_to_mesh
from src.loading import load_calibration

# Load camera calibration
calibration = load_calibration('.', 'cam_0', sequence=1)

# Project 3D world point to 2D image
world_point = [28.07, 180.42, -1.27]
pixel_coords = project_world_to_camera(world_point, calibration)

# Project 2D bounding box to 3D mesh
bbox = {'x': 807.0, 'y': 359.0, 'w': 13.0, 'h': 31.0}
world_coords = bbox_to_mesh(bbox, calibration, mesh)
```

### 🔥 PyTorch Integration (`src/pytorch_dataset.py`)
Ready-to-use PyTorch datasets:

```python
from src.loading import AnnotationDataset, collate_fn
from torch.utils.data import DataLoader

# Create PyTorch dataset
dataset = AnnotationDataset(loader, frame_ids, cam_ids)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, 
                       collate_fn=collate_fn, num_workers=4)

# Iterate through batches
for batch in dataloader:
    images = batch['images']        # Tensor of shape (B, C, H, W)
    bboxes = batch['bboxes']        # List of bounding boxes
    world_coords = batch['world']   # List of world coordinates
    track_ids = batch['track_ids']  # List of track IDs
```

## 📖 Usage Examples

### Basic Data Loading and Visualization
See [`example.ipynb`](example.ipynb) for:
- Loading annotations in different formats
- Visualizing 2D annotations and 3D world coordinates
- Working with camera calibration
- PyTorch dataset integration

### Evaluation and Metrics
See [`evaluation.ipynb`](evaluation.ipynb) for detailed examples of:
- **3D World Coordinate Evaluation**: Using `compute_mot_metric()` for evaluation in 3D space
- **Standard MOT Evaluation**: Using `MOTMetricEvaluator` for MOTA, MOTP, IDF1, and HOTA metrics
- **Synthetic Data Generation**: Creating test predictions with `simulate_predictions()`
- **Data Format Conversion**: Converting between Scout format and evaluation formats

## 🎯 Dataset Information

### Dataset Access
- **Main Website**: [https://scout.epfl.ch/](https://scout.epfl.ch/)
- **Research Group**: [EPFL Computer Vision Laboratory](https://www.epfl.ch/labs/cvlab/)

### Dataset Features
- **Multi-camera setup**: Synchronized cameras with overlapping fields of view
- **3D world coordinates**: Precise 3D localization of pedestrians
- **Multiple annotation formats**: COCO, Individual, and MOT format support
- **Camera calibration**: High-precision camera calibration parameters
- **Realistic scenarios**: Outdoor pedestrian tracking in challenging conditions

### Associated Research

- **Paper**: "Unified People Tracking with Graph Neural Networks"
- **Arxiv**: [https://arxiv.org/abs/2507.08494](https://arxiv.org/abs/2507.08494)
- **Code Repository**: [https://github.com/cvlab-epfl/umpn](https://github.com/cvlab-epfl/umpn)

## Citation

If you use this dataset in your research, please cite:


```bibtex
@article{engilberge25scout,
  title = {Unified People Tracking with Graph Neural Networks},
  author = {Martin Engilberge and Ivan Vrkic and Friedrich Wilke Grosche 
            and Julien Pilet and Engin Turetken and Pascal Fua},
  journal = {arXiv preprint arXiv:2507.08494},
  year = {2025}
}
```


## 📝 License

Please refer to the [official Scout dataset website](https://scout.epfl.ch/) for licensing information and usage terms.
