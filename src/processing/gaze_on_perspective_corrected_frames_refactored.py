"""
gaze_on_perspective_corrected_frames_refactored.py - Modular gaze processing with perspective correction

This script performs the core "Perspective Correction" step.
Raw eye tracking video records the scene from the user's head, which moves.
To analyze gaze patterns (e.g., "is the user looking at the top-left corner?"),
we must stabilize the video to a fixed reference frame.

We use **ArUco markers** (square QR-code-like tags) placed at the corners of the workspace.
By detecting these markers, we calculate a "Homography Matrix" that transforms the moving
camera view into a flat, top-down view.
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

    Why? Sometimes markers are hard to detect in poor lighting.
    Enhancing the image helps the ArUco detector find them.

    Args:
        frame: The original video frame (image).

    Returns:
        The enhanced frame.
    """
    # Convert to grayscale (simplifies processing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization spreads out the most frequent intensity values.
    # It increases global contrast.
    equalized = cv2.equalizeHist(gray)

    # Convert back to BGR color space (OpenCV uses BGR, not RGB)
    enhanced_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    # Apply a sharpening filter using a kernel convolution
    # This emphasizes edges, making marker borders crisper.
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_frame, -1, kernel)
    return sharpened


def apply_preselected_parameters(parameters):
    """
    Apply a predefined set of ArUco detector parameters.
    These are "tuned" values that work well for typical surgical videos.
    """
    # Thresholding logic: deciding what is black vs white
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 80
    parameters.adaptiveThreshWinSizeStep = 5
    
    # Filter out contours that are too small or too big to be valid markers
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 2
    
    # Sub-pixel refinement gives us more precise corner coordinates
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 25
    
    return parameters


def find_outer_points(corners, ids, valid_ids):
    """
    Find the most outer points from all marker corners.
    Instead of using the center of the markers, we use the furthest corners
    to define the largest possible workspace area.
    """
    all_points = []
    # Collect all corner points from valid markers
    for corner, marker_id in zip(corners, ids.flatten()):
        if marker_id in valid_ids:
            all_points.extend(corner[0])
    
    if len(all_points) == 0:
        return None
    
    all_points = np.array(all_points, dtype="float32")
    
    if all_points.shape[0] < 4:
        return None
    
    # convexHull finds the "rubber band" shape around the points
    hull = cv2.convexHull(all_points)
    
    # We need exactly 4 points to define a quadrilateral for perspective transform
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
    Gaze data is stored as a series of JSON objects in a .gz file.
    """
    with gzip.open(gaze_file_path, 'rt') as f:
        gaze_data = [json.loads(line) for line in f]
    return gaze_data


def extract_timestamps_and_gaze_positions(gaze_data):
    """
    Extract timestamps and gaze positions with alignment.

    Returns:
        timestamps: List of times in seconds.
        gaze_positions: List of [x, y] coordinates (0-1 normalized).
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
    Find the center point of each valid marker.
    This is the standard way to define the 4 corners of the workspace.
    """
    marker_points = []
    for corner, marker_id in zip(corners, ids.flatten()):
        if marker_id in valid_ids:
            # Calculate the mean (center) of the 4 corners of the marker
            avg_point = np.mean(corner[0], axis=0)
            marker_points.append(avg_point)

    # We strictly need 4 markers (IDs 13, 14, 15, 16) to define the plane
    if len(marker_points) != 4:
        return None
    return order_points(np.array(marker_points, dtype="float32"))


def _process_frame_chunk(args):
    """
    Worker function to process a chunk of video frames.

    Because video processing is slow, we split the video into "chunks"
    and process them in parallel on different CPU cores.
    """
    
    # --- Helper function for interpolating gaze ---
    def _interpolate_gaze(timestamps, gaze_positions, frame_time):
        """
        Gaze data (100Hz) is faster than video (30Hz).
        We need to find the gaze position exactly at the moment the frame was captured.
        We linear interpolate between the two nearest gaze samples.
        """
        # Binary search to find the index
        idx_before = np.searchsorted(timestamps, frame_time, side='right') - 1
        idx_after = idx_before + 1

        if idx_before < 0 or idx_after >= len(timestamps):
            return None

        t_before, t_after = timestamps[idx_before], timestamps[idx_after]
        gaze_before, gaze_after = gaze_positions[idx_before], gaze_positions[idx_after]

        # Interpolation formula: P = P1 + alpha * (P2 - P1)
        if any(np.isnan(gaze_before)) or any(np.isnan(gaze_after)):
            return None

        if t_after == t_before:
            return np.array(gaze_before)

        alpha = (frame_time - t_before) / (t_after - t_before)
        return (1 - alpha) * np.array(gaze_before) + alpha * np.array(gaze_after)

    # --- Unpack arguments ---
    (video_path, chunk_index, start_frame, end_frame, gaze_data,
     frame_width, frame_height, output_width, output_height,
     target_markers, use_preselected_parameters, use_frame_preprocessing,
     use_outer_points) = args

    # --- Initialize resources ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Jump to the start of our chunk
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    timestamps, gaze_positions = extract_timestamps_and_gaze_positions(gaze_data)

    # --- Initialize ArUco detector ---
    # DICT_4X4_50 means 4x4 grid markers, 50 possible IDs. Standard for this project.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    if use_preselected_parameters:
        parameters = apply_preselected_parameters(parameters)
    
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    except AttributeError:
        detector = None # Compatibility fallback

    # --- Local storage for this chunk ---
    chunk_video_frames = []
    chunk_csv_data = []
    chunk_transformation_history = []
    persistent_homography = None # Remember the last valid matrix if markers disappear briefly

    stats = {
        'total_frames': 0, 'frames_with_markers': 0,
        'frames_with_valid_homography': 0, 'frames_with_gaze': 0
    }

    # --- Process loop ---
    frame_range = range(start_frame, end_frame)
    if end_frame - start_frame > 1000:
        frame_range = tqdm(frame_range, desc=f"Processing frames", leave=False)

    for frame_index in frame_range:
        ret, frame = cap.read()
        if not ret:
            break

        stats['total_frames'] += 1
        frame_time = frame_index / fps

        # 1. Preprocess
        preprocessed_frame = enhance_contrast_and_sharpness(frame) if use_frame_preprocessing else frame
        gray = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)

        # 2. Detect Markers
        if detector:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # 3. Calculate Homography
        # We need to find the transform that maps the detected markers (src)
        # to the corners of our output video (dst).
        if ids is not None and len(ids) > 0:
            stats['frames_with_markers'] += 1
            target_markers_set = set(target_markers)

            if target_markers_set.issubset(set(ids.flatten())):
                # Get the 4 corners from the detected markers
                src_points = find_outer_points(corners, ids, list(target_markers_set)) if use_outer_points else find_and_order_average_points(corners, ids, list(target_markers_set))

                if src_points is not None:
                    # Define destination points (the corners of the output video)
                    desired_corners = np.array([
                        [0, 0],
                        [output_width - 1, 0],
                        [output_width - 1, output_height - 1],
                        [0, output_height - 1]
                    ], dtype="float32")

                    # Compute Homography Matrix H
                    H, _ = cv2.findHomography(src_points, desired_corners)
                    persistent_homography = H # Save it
                    stats['frames_with_valid_homography'] += 1

        # 4. Save Transformation (for later use in CSV generation)
        chunk_transformation_history.append({
            'frame_index': frame_index,
            'frame_time': round(frame_time, 3),
            'homography_matrix': persistent_homography.copy() if persistent_homography is not None else None
        })

        # 5. Apply Transformation to Video and Gaze
        transformed_gaze_x, transformed_gaze_y = np.nan, np.nan

        if persistent_homography is not None:
            # Warp the video image ("un-tilt" it)
            corrected_frame = cv2.warpPerspective(frame, persistent_homography, (output_width, output_height))

            # Find gaze point for this time
            interpolated_gaze = _interpolate_gaze(timestamps, gaze_positions, frame_time)

            if interpolated_gaze is not None:
                # Apply the SAME transformation to the gaze point
                transformed_gaze = transform_gaze_point(interpolated_gaze, persistent_homography, frame_width, frame_height)
                if transformed_gaze:
                    tx, ty = transformed_gaze
                    if 0 <= tx < output_width and 0 <= ty < output_height:
                        transformed_gaze_x, transformed_gaze_y = tx, ty
                        stats['frames_with_gaze'] += 1
                        # Draw gaze circle on video
                        cv2.circle(corrected_frame, (int(tx), int(ty)), 15, (0, 0, 255), -1)

            chunk_video_frames.append(corrected_frame)
        else:
            # If we've never seen markers, output black frame
            blank_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            chunk_video_frames.append(blank_frame)

        # 6. Save data for CSV
        chunk_csv_data.append([frame_index, round(frame_time, 3), transformed_gaze_x, transformed_gaze_y])

    cap.release()
    return (chunk_index, chunk_video_frames, chunk_csv_data, chunk_transformation_history, stats)


def process_gaze_with_perspective_correction(
    video_path, gaze_file_path, output_video_path, csv_output_path,
    transformation_history_path, output_width=1000, output_height=606,
    target_markers=None, use_preselected_parameters=False, use_frame_preprocessing=False,
    use_outer_points=False, show_video=False, num_workers=None
):
    """
    Main entry function for perspective correction.
    Orchestrates parallel processing of the video.
    """
    if target_markers is None:
        target_markers = [13, 14, 15, 16]
    
    # Setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file at {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")
    
    gaze_data = load_gaze_data(gaze_file_path)
    
    # Parallel setup
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # Calculate chunks
    chunk_size = math.ceil(total_frames / num_workers)
    chunks = [(i, i * chunk_size, min((i + 1) * chunk_size, total_frames)) for i in range(num_workers)]
    
    logger.info(f"Starting processing with {num_workers} workers...")

    processing_args = [
        (video_path, chunk_index, start, end, gaze_data,
         frame_width, frame_height, output_width, output_height,
         target_markers, use_preselected_parameters, use_frame_preprocessing, use_outer_points)
        for chunk_index, start, end in chunks
    ]

    # Execute
    if num_workers == 1:
        results = [_process_frame_chunk(args) for args in tqdm(processing_args, desc="Processing chunks")]
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_process_frame_chunk, processing_args), total=len(chunks), desc="Processing chunks"))

    # Sort results to reassemble video in order
    results.sort(key=lambda x: x[0])

    # Aggregate results
    logger.info("Aggregating results...")
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
            
    # Write outputs
    
    # 1. Save Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    for frame in tqdm(all_video_frames, desc="Writing video"):
        out.write(frame)
    out.release()

    # 2. Save CSV
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_index', 'frame_time', 'transformed_gaze_x', 'transformed_gaze_y'])
        writer.writerows(all_csv_data)

    # 3. Save Matrix History (Critical for Step 2)
    all_transformation_history.sort(key=lambda x: x['frame_index'])
    np.save(transformation_history_path, all_transformation_history, allow_pickle=True)
    
    logger.info(f"Complete. Valid frames: {total_stats['frames_with_valid_homography']}/{total_stats['total_frames']}")
    return total_stats
