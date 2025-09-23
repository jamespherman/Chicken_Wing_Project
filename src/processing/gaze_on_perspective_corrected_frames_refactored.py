"""
gaze_on_perspective_corrected_frames_refactored.py - Modular gaze processing with perspective correction

"""

import cv2
import numpy as np
import os
import gzip
import json
import csv
from tqdm import tqdm
from .utils import order_points, transform_gaze_point
import multiprocessing
import math
from ..logging_config import get_logger

logger = get_logger(__name__)


############################################
# Preprocessing toggle features
############################################

def enhance_contrast_and_sharpness(frame):
    """
    Enhance contrast and sharpness of the input frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhanced_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_frame, -1, kernel)
    return sharpened


def apply_preselected_parameters(parameters):
    """
    Apply a predefined set of ArUco detector parameters.
    """
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 80
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.adaptiveThreshConstant = 0
    
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 2
    
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.minCornerDistanceRate = 0.18
    
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 25
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01
    
    return parameters


def find_outer_points(corners, ids, valid_ids):
    """
    Find the most outer points from all marker corners.
    """
    all_points = []
    for corner, marker_id in zip(corners, ids.flatten()):
        if marker_id in valid_ids:
            all_points.extend(corner[0])
    
    if len(all_points) == 0:
        return None
    
    all_points = np.array(all_points, dtype="float32")
    
    if all_points.shape[0] < 4:
        return None
    
    hull = cv2.convexHull(all_points)
    
    if len(hull) >= 4:
        ordered_points = order_points(hull.squeeze())
        return ordered_points
    return None


############################################
# Gaze processing functions
############################################

def load_gaze_data(gaze_file_path):
    """
    Load gaze data from compressed file.
    """
    with gzip.open(gaze_file_path, 'rt') as f:
        gaze_data = [json.loads(line) for line in f]
    return gaze_data


def extract_timestamps_and_gaze_positions(gaze_data):
    """
    Extract timestamps and gaze positions with alignment.
    """
    timestamps = []
    gaze_positions = []
    for sample in gaze_data:
        timestamps.append(sample['timestamp'])
        if 'gaze2d' in sample['data']:
            gaze_positions.append(sample['data']['gaze2d'])
        else:
            gaze_positions.append([np.nan, np.nan])
    return timestamps, gaze_positions


def find_and_order_average_points(corners, ids, valid_ids):
    """
    Find and order average points from marker corners.
    """
    marker_points = []
    for corner, marker_id in zip(corners, ids.flatten()):
        if marker_id in valid_ids:
            avg_point = np.mean(corner[0], axis=0)
            marker_points.append(avg_point)
    if len(marker_points) != 4:
        return None
    return order_points(np.array(marker_points, dtype="float32"))


def _process_frame_chunk(args):
    """
    Worker function to process a chunk of video frames.

    This function is designed to be called by a multiprocessing pool. It takes a single
    tuple argument to be compatible with `pool.starmap`.
    """
    
    # --- Helper functions (re-defined in worker for encapsulation) ---

    def _interpolate_gaze(timestamps, gaze_positions, frame_time):
        idx_before = np.searchsorted(timestamps, frame_time, side='right') - 1
        idx_after = idx_before + 1

        if idx_before < 0 or idx_after >= len(timestamps):
            return None

        t_before, t_after = timestamps[idx_before], timestamps[idx_after]
        gaze_before, gaze_after = gaze_positions[idx_before], gaze_positions[idx_after]

        if any(np.isnan(gaze_before)) or any(np.isnan(gaze_after)):
            return None

        # Avoid division by zero if timestamps are identical
        if t_after == t_before:
            return np.array(gaze_before)

        alpha = (frame_time - t_before) / (t_after - t_before)
        return (1 - alpha) * np.array(gaze_before) + alpha * np.array(gaze_after)

    # --- Unpack arguments ---
    (video_path, chunk_index, start_frame, end_frame, gaze_data,
     frame_width, frame_height, output_width, output_height,
     target_markers, use_preselected_parameters, use_frame_preprocessing,
     use_outer_points) = args

    # --- Initialize resources for this worker ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    timestamps, gaze_positions = extract_timestamps_and_gaze_positions(gaze_data)

    # --- Initialize ArUco detector ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    if use_preselected_parameters:
        parameters = apply_preselected_parameters(parameters)
    
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    except AttributeError:
        detector = None # Will use detectMarkers directly

    # --- Initialize local variables for this chunk ---
    chunk_video_frames = []
    chunk_csv_data = []
    chunk_transformation_history = []
    persistent_homography = None

    stats = {
        'total_frames': 0, 'frames_with_markers': 0,
        'frames_with_valid_homography': 0, 'frames_with_gaze': 0
    }

    # --- Process frames in the assigned chunk ---
    for frame_index in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        stats['total_frames'] += 1
        frame_time = frame_index / fps

        # --- Frame preprocessing and marker detection ---
        preprocessed_frame = enhance_contrast_and_sharpness(frame) if use_frame_preprocessing else frame
        gray = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)

        if detector:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # --- Homography calculation ---
        if ids is not None and len(ids) > 0:
            stats['frames_with_markers'] += 1
            target_markers_set = set(target_markers)

            if target_markers_set.issubset(set(ids.flatten())):
                src_points = find_outer_points(corners, ids, list(target_markers_set)) if use_outer_points else find_and_order_average_points(corners, ids, list(target_markers_set))

                if src_points is not None:
                    desired_corners = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype="float32")
                    H, _ = cv2.findHomography(src_points, desired_corners)
                    persistent_homography = H
                    stats['frames_with_valid_homography'] += 1

        # --- Record transformation history ---
        chunk_transformation_history.append({
            'frame_index': frame_index,
            'frame_time': round(frame_time, 3),
            'homography_matrix': persistent_homography.copy() if persistent_homography is not None else None
        })

        # --- Gaze processing and video output ---
        transformed_gaze_x, transformed_gaze_y = np.nan, np.nan

        if persistent_homography is not None:
            corrected_frame = cv2.warpPerspective(frame, persistent_homography, (output_width, output_height))
            interpolated_gaze = _interpolate_gaze(timestamps, gaze_positions, frame_time)

            if interpolated_gaze is not None:
                transformed_gaze = transform_gaze_point(interpolated_gaze, persistent_homography, frame_width, frame_height)
                if transformed_gaze:
                    tx, ty = transformed_gaze
                    if 0 <= tx < output_width and 0 <= ty < output_height:
                        transformed_gaze_x, transformed_gaze_y = tx, ty
                        stats['frames_with_gaze'] += 1
                        cv2.circle(corrected_frame, (tx, ty), 15, (0, 0, 255), -1)

            chunk_video_frames.append(corrected_frame)
        else:
            # If no valid homography, add a blank frame to maintain video sync
            blank_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            chunk_video_frames.append(blank_frame)

        # --- CSV data ---
        chunk_csv_data.append([frame_index, round(frame_time, 3), transformed_gaze_x, transformed_gaze_y])

    # --- Cleanup and return results for this chunk ---
    cap.release()
    return (chunk_index, chunk_video_frames, chunk_csv_data, chunk_transformation_history, stats)


def process_gaze_with_perspective_correction(
    video_path, gaze_file_path, output_video_path, csv_output_path,
    transformation_history_path, output_width=1000, output_height=606,
    target_markers=None, use_preselected_parameters=False, use_frame_preprocessing=False,
    use_outer_points=False, show_video=False, num_workers=None
):
    """
    Process video with gaze data and perspective correction in parallel.
    """
    if target_markers is None:
        target_markers = [13, 14, 15, 16]
    
    # --- Video properties and setup ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file at {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Video properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")
    
    # --- Load gaze data (once) ---
    gaze_data = load_gaze_data(gaze_file_path)
    logger.info(f"Loaded {len(gaze_data)} gaze samples")
    
    # --- Parallel processing setup ---
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    chunk_size = math.ceil(total_frames / num_workers)
    chunks = [(i, i * chunk_size, min((i + 1) * chunk_size, total_frames)) for i in range(num_workers)]
    
    logger.info(f"Starting parallel processing with {num_workers} workers, {len(chunks)} chunks...")
    
    # --- Prepare arguments for each worker ---
    processing_args = [
        (video_path, chunk_index, start, end, gaze_data,
         frame_width, frame_height, output_width, output_height,
         target_markers, use_preselected_parameters, use_frame_preprocessing, use_outer_points)
        for chunk_index, start, end in chunks
    ]
    
    # --- Run multiprocessing pool ---
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(_process_frame_chunk, processing_args), total=len(chunks), desc="Processing chunks"))

    # --- Sort results by chunk index to maintain order ---
    results.sort(key=lambda x: x[0])

    # --- Aggregate results ---
    logger.info("Aggregating results from all workers...")
    all_video_frames = []
    all_csv_data = []
    all_transformation_history = []
    total_stats = {
        'total_frames': 0, 'frames_with_markers': 0,
        'frames_with_valid_homography': 0, 'frames_with_gaze': 0
    }

    for _, video_frames, csv_data, trans_hist, stats in results:
        all_video_frames.extend(video_frames)
        all_csv_data.extend(csv_data)
        all_transformation_history.extend(trans_hist)
        for key in total_stats:
            total_stats[key] += stats[key]
            
    # --- Write outputs (sequentially) ---
    
    # 1. Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    for frame in tqdm(all_video_frames, desc="Writing video"):
        out.write(frame)
    out.release()

    # 2. CSV
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_index', 'frame_time', 'transformed_gaze_x', 'transformed_gaze_y'])
        writer.writerows(all_csv_data)

    # 3. Transformation History
    all_transformation_history.sort(key=lambda x: x['frame_index']) # Ensure sorted by frame index
    np.save(transformation_history_path, all_transformation_history, allow_pickle=True)
    
    total_stats['transformation_history_length'] = len(all_transformation_history)
    
    # --- Final report ---
    logger.info("Processing complete!")
    logger.info(f"Video saved to: {output_video_path}")
    logger.info(f"Gaze data saved to: {csv_output_path}")
    logger.info(f"Transformation history saved to: {transformation_history_path}")
    logger.info(f"Total frames processed: {total_stats['total_frames']}")
    logger.info(f"Frames with markers: {total_stats['frames_with_markers']}")
    logger.info(f"Frames with valid homography: {total_stats['frames_with_valid_homography']}")
    logger.info(f"Frames with gaze points: {total_stats['frames_with_gaze']}")

    return total_stats


