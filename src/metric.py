from pathlib import Path
import shutil
import numpy as np
import torch
import pandas as pd

import trackeval
import motmetrics as mm

import random


def make_dataframe_from_det(det_list):
    det_as_list = list()
    for frame_id, frame_det in enumerate(det_list):
        det_as_list.extend([{'FrameId':frame_id, 'Id':-1, 'X':int(det[0]), 'Y':int(det[1])} for det in frame_det])

    det_as_df = pd.DataFrame(det_as_list)

    if  det_as_df.empty:
        det_as_df = pd.DataFrame(columns =['FrameId','Id','X','Y'])

    det_as_df = det_as_df.set_index(['FrameId', 'Id'])

    return det_as_df


def compute_mot_metric(gt_df, pred_df, metric_threshold, nb_gt):

    if gt_df.size == 0:
        print("Trying to compute tracking metric on an empty sequence (gt size is 0)")
        return None

    acc = mm.utils.compare_to_groundtruth(gt_df, pred_df, 'euc', distfields=['X', 'Y'], distth=metric_threshold)
    
    #library doesn't implement moda computation, compute it manually form accumulator
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')

    # print(summary)  # Uncomment for debugging
    metrics = dict(zip(summary.keys(), summary.values[0]))
    
    return metrics



class MOTMetricEvaluator:
    def __init__(self, interpolate_missing_detections=False):
        self.initialized = False
        self.sequences = set()
        
        self.interpolate_missing_detections = interpolate_missing_detections
        
    def _initialize_directories(self, dset_name):
        """Initialize directory structure for trackeval"""
        self.root_track_eval_path = Path("TrackEvalDataTemp") / f"GNN_{dset_name}"
        self.root_track_eval_path.mkdir(exist_ok=True, parents=True)

        
        # Clear any existing files
        for item in self.root_track_eval_path.glob('*'):
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(str(item))
                
        # Create seqmap file with header
        seqmap_path = self.root_track_eval_path / f"gt/seqmaps/{dset_name}-seqmaps.txt"
        seqmap_path.parent.mkdir(parents=True, exist_ok=True)
        with open(seqmap_path, 'w') as f:
            f.write("name\n")
                
        self.initialized = True

    def save_sequence_data(self, pred_bboxes, pred_world_points, pred_timestamps, pred_ids, gt_dict, dset_name, sequence):
        """
        Save ground truth and predictions for a single sequence
        
        Args:
            pred_bboxes: torch.Tensor of shape (N, 4) - predicted bboxes in (x1, y1, x2, y2) format
            pred_world_points: torch.Tensor of shape (N, 3) - predicted world coordinates
            pred_timestamps: torch.Tensor of shape (N,) - timestamps for each prediction
            pred_ids: torch.Tensor of shape (N,) - predicted track IDs
            gt_dict: dict - ground truth data in the expected format
            dset_name: str - dataset name
            sequence: str - sequence name
        """

        if not self.initialized:
            self._initialize_directories(dset_name)
            
        self.sequences.add((dset_name, sequence))
        
        if dset_name == "MOT17":
            detector = "-DPM"
        else:
            detector = ""

        print(f"Saving tracking data for {dset_name} {sequence}")

        # Prepare data structures for saving predictions and ground truth
        track_as_list_pred = []
        track_as_list_gt = []
        
        # Process each timestamp
        timestamps = sorted(gt_dict.keys())
        min_timestamp = min(timestamps)-1
        
        # Lists to store gt and pred info
        gt_world_points = []
        gt_track_ids = []
        gt_timestamps = []
        pred_world_points_list = []
        pred_track_ids_list = []
        pred_timestamps_list = []

        timestamp_to_skip = []
        frame_id = 1
        for timestamp in timestamps:
            # Convert timestamp to 0-based index
            # frame_id = int(timestamp - min_timestamp)
            
            skip_timestamp = False
            for view_id, view_data in gt_dict[timestamp].items():
                if len(view_data['bbox']) == 0:
                    # Add empty gt detection with confiddence 0 to validate the itmestamp
                    # track_as_list_gt.append((frame_id, -1, 0, 0, 0, 0, 0, 0, 0))
                    skip_timestamp = True
                    continue
                
                for bbox, person_id, world_point in zip(view_data['bbox'], view_data['person_id'], view_data['world_points']):
                    track_as_list_gt.append((
                        frame_id,
                        int(person_id), 
                        float(bbox[0]), float(bbox[1]),
                        float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1]),
                        1, 1, 1
                    ))
                    gt_world_points.append(torch.tensor(world_point))
                    gt_track_ids.append(torch.tensor(person_id))
                    gt_timestamps.append(torch.tensor(timestamp))
            
            if skip_timestamp:
                continue
            
            # Prediction data
            mask = pred_timestamps == timestamp
            if mask.any():
                batch_pred_bboxes = pred_bboxes[mask]
                batch_pred_track_ids = pred_ids[mask]
                batch_pred_world_points = pred_world_points[mask]
                
                # Filter out detections with no trajectory
                valid_pred_mask = batch_pred_track_ids != -1
                batch_pred_bboxes = batch_pred_bboxes[valid_pred_mask]
                batch_pred_track_ids = batch_pred_track_ids[valid_pred_mask]
                batch_pred_world_points = batch_pred_world_points[valid_pred_mask]
                
                for bbox, track_id, world_point in zip(batch_pred_bboxes, batch_pred_track_ids, batch_pred_world_points):
                    track_as_list_pred.append({
                        'FrameId': frame_id,
                        'Id': int(track_id)+1,
                        'X': float(bbox[0]),
                        'Y': float(bbox[1]), 
                        'Width': float(bbox[2]-bbox[0]),
                        'Height': float(bbox[3]-bbox[1]),
                        'Score': 1.0
                    })
                    pred_world_points_list.append(world_point)
                    pred_track_ids_list.append(track_id)
                    pred_timestamps_list.append(torch.tensor(timestamp))
            
            frame_id += 1

        if self.interpolate_missing_detections:
            # Note: interpolate_missing_detections function not implemented
            # track_as_list_pred = interpolate_missing_detections(track_as_list_pred)
            print("Warning: interpolate_missing_detections not implemented, skipping")

        # Save predictions
        pred_path = self.root_track_eval_path / f"pred/{dset_name}-seqmaps/GNNTracker/data/{dset_name}-{sequence}.txt"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, 'w') as f:
            for pred in track_as_list_pred:
                f.write(f"{pred['FrameId']},{pred['Id']},{pred['X']},{pred['Y']},{pred['Width']},{pred['Height']},{pred['Score']},-1,-1,-1\n")

        # Save ground truth
        gt_path = self.root_track_eval_path / f"gt/{dset_name}-seqmaps/{dset_name}-{sequence}/gt/gt.txt"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, 'w') as f:
            for gt in track_as_list_gt:
                f.write(f"{gt[0]},{gt[1]},{gt[2]},{gt[3]},{gt[4]},{gt[5]},{gt[6]},{gt[7]},{gt[8]}\n")

        # Read and save seqinfo.ini
        path_org_seqinfo = Path(f"/cvlabscratch/cvlab/home/engilber/datasets/{dset_name}/train/{dset_name}-{sequence}{detector}/seqinfo.ini")
        
        if path_org_seqinfo.exists():
            with open(path_org_seqinfo, 'r') as file:
                data = file.readlines()
        else:
            print(f"Warning: No seqinfo.ini file found for {dset_name} {sequence}, using default values")
            data = [
                "[Sequence]\n",
                f"name={sequence}\n",
                "imDir=img1\n",
                "frameRate=30\n", 
                "seqLength=1\n",
                "imWidth=1920\n",
                "imHeight=1080\n",
                "imExt=.jpg\n"
            ]

        seqinfo_path = self.root_track_eval_path / f"gt/{dset_name}-seqmaps/{dset_name}-{sequence}/seqinfo.ini"
        seqinfo_path.parent.mkdir(parents=True, exist_ok=True)

        # Update seqLength in seqinfo
        new_seq_length = max(t[0] for t in track_as_list_gt)#len(set(t[0] for t in track_as_list_gt))
        new_data = []
        for line in data:
            if "seqLength" in line:
                new_data.append(f"seqLength={new_seq_length}\n")
            else:
                new_data.append(line)

        with open(seqinfo_path, 'w') as file:
            file.writelines(new_data)

        # Append sequence to seqmap
        seqmap_path = self.root_track_eval_path / f"gt/seqmaps/{dset_name}-seqmaps.txt"
        with open(seqmap_path, 'a') as f:
            f.write(f"{dset_name}-{sequence}\n")

        # Concatenate all timesteps
        gt_world_points = torch.stack(gt_world_points) if gt_world_points else torch.empty((0, 3))
        gt_track_ids = torch.stack(gt_track_ids) if gt_track_ids else torch.empty(0)
        gt_timestamps = torch.stack(gt_timestamps) if gt_timestamps else torch.empty(0)

        if len(pred_world_points_list) > 0:
            pred_world_points_out = torch.stack(pred_world_points_list)
            pred_track_ids_out = torch.stack(pred_track_ids_list)
            pred_timestamps_out = torch.stack(pred_timestamps_list)
        else:
            pred_world_points_out = torch.empty((0, 3))
            pred_track_ids_out = torch.empty(0)
            pred_timestamps_out = torch.empty(0)

        return (gt_world_points, gt_track_ids, gt_timestamps), (pred_world_points_out, pred_track_ids_out, pred_timestamps_out)

    def compute_metrics(self, dset_name):
        """Compute metrics for all saved sequences"""
        # Configure evaluation
        eval_config = trackeval.Evaluator.get_default_eval_config()
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5, 'PRINT_CONFIG': False}

        eval_config.update({
            'PRINT_ONLY_COMBINED': False,
            'PRINT_DETAILED': False,
            'PRINT_SUMMARY': False,
            'PRINT_CONFIG': False,
            'TIME_PROGRESS': False,
            'LOG_ON_ERROR': str(self.root_track_eval_path / "error.log"),
            'OUTPUT_DETAILED': False,
            'PRINT_RESULTS': False,
            'OUTPUT_SUMMARY': False,
            'PRINT_ONLY_COMBINED': True,
            'DISPLAY_LESS_PROGRESS': True,
            'PLOT_CURVES': False,
            'PRINT_CONFIG': False
        })

        dataset_config.update({
            'GT_FOLDER': str(self.root_track_eval_path / "gt"),
            'TRACKERS_FOLDER': str(self.root_track_eval_path / "pred"),
            'BENCHMARK': dset_name,
            'SPLIT_TO_EVAL': 'seqmaps',
            'TRACKERS_TO_EVAL': ['GNNTracker'],
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 1,
            'PRINT_CONFIG': False
        })

        # Run evaluation
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric(metrics_config))

        output_res, _ = evaluator.evaluate(dataset_list, metrics_list)

                # Flatten metrics for each sequence
        sequence_metrics = {}
        for seq_name, seq_data in output_res['MotChallenge2DBox']['GNNTracker'].items():
            flattened_metrics = {}
            for category in seq_data['pedestrian']:
                for metric, value in seq_data['pedestrian'][category].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) or metric == 'HOTA':
                        if metric == 'HOTA':
                            value = np.mean(value)
                        if seq_name == 'COMBINED_SEQ':
                            flattened_metrics[f'combined_{metric}'] = value
                        else:
                            flattened_metrics[f'{seq_name}_{metric}'] = value
            sequence_metrics.update(flattened_metrics)

        # Clean up temporary files if needed
        # shutil.rmtree(self.root_track_eval_path)

        return sequence_metrics


def scout_to_dataframe(sequence_data, use_world_coords=True):
    """
    Convert Scout annotation format to dataframe format expected by compute_mot_metric.
    
    Args:
        sequence_data: Dictionary with frame_id -> list of annotations
        use_world_coords: If True, use world coordinates; if False, use bbox center
    
    Returns:
        pandas.DataFrame with columns ['FrameId', 'Id', 'X', 'Y']
    """
    data_list = []
    
    for frame_id, frame_anns in sequence_data.items():
        for ann in frame_anns:
            if use_world_coords:
                # Use world coordinates (Xw, Yw)
                x = ann['world']['Xw']
                y = ann['world']['Yw']
            else:
                # Use bbox center coordinates
                x = ann['bbox']['x'] + ann['bbox']['w'] / 2
                y = ann['bbox']['y'] + ann['bbox']['h'] / 2
            
            data_list.append({
                'FrameId': frame_id,
                'Id': ann['track_id'], 
                'X': x,
                'Y': y
            })
    
    df = pd.DataFrame(data_list)
    if df.empty:
        df = pd.DataFrame(columns=['FrameId', 'Id', 'X', 'Y'])
    
    return df.set_index(['FrameId', 'Id'])


def simulate_predictions(gt_sequence, noise_params=None):
    """
    Create simulated predictions by modifying ground truth with various types of noise.
    Ensures unique track IDs within each frame to avoid trackeval errors.
    
    Args:
        gt_sequence: Dictionary with frame_id -> list of annotations
        noise_params: Dictionary of noise parameters
    
    Returns:
        pred_sequence: Dictionary with frame_id -> list of predictions
    """
    if noise_params is None:
        noise_params = {
            'delete_prob': 0.1,     # Probability of missing a detection
            'shift_prob': 0.3,      # Probability of bbox/position shift  
            'id_switch_prob': 0.05, # Probability of ID switch
            'false_positive_prob': 0.05,  # Probability of false positive
            'bbox_noise_std': 5.0,  # Standard deviation for bbox noise
            'world_noise_std': 0.5  # Standard deviation for world coordinate noise
        }
    
    pred_sequence = {}
    all_track_ids = set()
    
    # Collect all track IDs for ID switching
    for frame_anns in gt_sequence.values():
        for ann in frame_anns:
            all_track_ids.add(ann['track_id'])
    all_track_ids = list(all_track_ids)
    
    # Counter for generating unique false positive IDs
    fp_id_counter = max(all_track_ids) + 1000
    
    for frame_id, frame_anns in gt_sequence.items():
        pred_frame = []
        used_ids_in_frame = set()  # Track IDs used in this frame to avoid duplicates
        
        for ann in frame_anns:
            # Skip detection (miss)
            if random.random() < noise_params['delete_prob']:
                continue
                
            # Copy annotation
            pred_ann = {
                'track_id': ann['track_id'],
                'bbox': ann['bbox'].copy(),
                'world': ann['world'].copy()
            }
            
            # Add bbox noise
            if random.random() < noise_params['shift_prob']:
                pred_ann['bbox']['x'] += random.gauss(0, noise_params['bbox_noise_std'])
                pred_ann['bbox']['y'] += random.gauss(0, noise_params['bbox_noise_std'])
                pred_ann['bbox']['w'] *= random.uniform(0.8, 1.2)
                pred_ann['bbox']['h'] *= random.uniform(0.8, 1.2)
            
            # Add world coordinate noise
            if random.random() < noise_params['shift_prob']:
                pred_ann['world']['Xw'] += random.gauss(0, noise_params['world_noise_std'])
                pred_ann['world']['Yw'] += random.gauss(0, noise_params['world_noise_std'])
                pred_ann['world']['Zw'] += random.gauss(0, noise_params['world_noise_std'] * 0.1)
            
            # ID switch - ensure no duplicates within frame
            if random.random() < noise_params['id_switch_prob']:
                # Try to find an unused ID from available IDs
                available_ids = [tid for tid in all_track_ids if tid not in used_ids_in_frame]
                if available_ids:
                    pred_ann['track_id'] = random.choice(available_ids)
                # If all IDs are used, create a new unique ID
                else:
                    pred_ann['track_id'] = fp_id_counter
                    fp_id_counter += 1
            
            # Check for duplicate IDs and resolve
            while pred_ann['track_id'] in used_ids_in_frame:
                pred_ann['track_id'] = fp_id_counter
                fp_id_counter += 1
            
            used_ids_in_frame.add(pred_ann['track_id'])
            pred_frame.append(pred_ann)
        
        # Add false positives
        if random.random() < noise_params['false_positive_prob']:
            # Create a false positive based on existing detection
            if frame_anns:
                base_ann = random.choice(frame_anns)
                fp_ann = {
                    'track_id': fp_id_counter,  # Use unique counter
                    'bbox': {
                        'x': base_ann['bbox']['x'] + random.gauss(0, 50),
                        'y': base_ann['bbox']['y'] + random.gauss(0, 50),
                        'w': base_ann['bbox']['w'] * random.uniform(0.5, 1.5),
                        'h': base_ann['bbox']['h'] * random.uniform(0.5, 1.5)
                    },
                    'world': {
                        'Xw': base_ann['world']['Xw'] + random.gauss(0, 2),
                        'Yw': base_ann['world']['Yw'] + random.gauss(0, 2),
                        'Zw': base_ann['world']['Zw'] + random.gauss(0, 0.5)
                    }
                }
                fp_id_counter += 1
                pred_frame.append(fp_ann)
        
        pred_sequence[frame_id] = pred_frame
    
    return pred_sequence


def validate_unique_ids(sequence_data):
    """
    Validate that there are no duplicate track IDs within any frame.
    
    Args:
        sequence_data: Dictionary with frame_id -> list of annotations
        
    Returns:
        bool: True if all IDs are unique within frames, False otherwise
    """
    for frame_id, frame_anns in sequence_data.items():
        ids_in_frame = [ann['track_id'] for ann in frame_anns]
        if len(ids_in_frame) != len(set(ids_in_frame)):
            duplicates = [id for id in set(ids_in_frame) if ids_in_frame.count(id) > 1]
            print(f"Frame {frame_id} has duplicate IDs: {duplicates}")
            return False
    return True


def convert_scout_to_tensors(gt_sequence, pred_sequence):
    """
    Convert Scout annotation format to tensors expected by MOTMetricEvaluator.
    
    Args:
        gt_sequence: Dictionary with frame_id -> list of annotations (ground truth)
        pred_sequence: Dictionary with frame_id -> list of annotations (predictions)
        
    Returns:
        tuple: (gt_dict, pred_bboxes, pred_world_points, pred_timestamps, pred_ids)
    """
    
    # Convert ground truth to MOTMetricEvaluator format
    gt_dict = {}
    
    for frame_id, frame_anns in gt_sequence.items():
        # Use frame_id as timestamp
        timestamp = frame_id
        
        if timestamp not in gt_dict:
            gt_dict[timestamp] = {}
        
        # Assume single view (view_id = 0) for monocular case
        view_id = 0
        gt_dict[timestamp][view_id] = {
            'bbox': [],
            'person_id': [],
            'world_points': []
        }
        
        for ann in frame_anns:
            # Convert bbox from (x, y, w, h) to (x1, y1, x2, y2)
            bbox = [
                ann['bbox']['x'],
                ann['bbox']['y'], 
                ann['bbox']['x'] + ann['bbox']['w'],
                ann['bbox']['y'] + ann['bbox']['h']
            ]
            
            world_point = [
                ann['world']['Xw'],
                ann['world']['Yw'], 
                ann['world']['Zw']
            ]
            
            gt_dict[timestamp][view_id]['bbox'].append(bbox)
            gt_dict[timestamp][view_id]['person_id'].append(ann['track_id'])
            gt_dict[timestamp][view_id]['world_points'].append(world_point)
    
    # Convert predictions to tensor format
    pred_bboxes_list = []
    pred_world_points_list = []
    pred_timestamps_list = []
    pred_ids_list = []
    
    for frame_id, frame_anns in pred_sequence.items():
        timestamp = frame_id
        
        for ann in frame_anns:
            # Convert bbox format
            bbox = torch.tensor([
                ann['bbox']['x'],
                ann['bbox']['y'],
                ann['bbox']['x'] + ann['bbox']['w'], 
                ann['bbox']['y'] + ann['bbox']['h']
            ])
            
            world_point = torch.tensor([
                ann['world']['Xw'],
                ann['world']['Yw'],
                ann['world']['Zw']
            ])
            
            pred_bboxes_list.append(bbox)
            pred_world_points_list.append(world_point)
            pred_timestamps_list.append(timestamp)
            pred_ids_list.append(ann['track_id'])
    
    # Convert to tensors
    if pred_bboxes_list:
        pred_bboxes = torch.stack(pred_bboxes_list)
        pred_world_points = torch.stack(pred_world_points_list)
        pred_timestamps = torch.tensor(pred_timestamps_list)
        pred_ids = torch.tensor(pred_ids_list)
    else:
        pred_bboxes = torch.empty((0, 4))
        pred_world_points = torch.empty((0, 3))
        pred_timestamps = torch.empty(0)
        pred_ids = torch.empty(0)
    
    return gt_dict, pred_bboxes, pred_world_points, pred_timestamps, pred_ids