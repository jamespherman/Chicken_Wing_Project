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
from utils import order_points


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


def transform_gaze_point(gaze_point, homography_matrix, frame_width, frame_height):
    """
    Apply homography to gaze point.
    """
    if any(np.isnan(gaze_point)):
        return None
    
    gaze_x = int(gaze_point[0] * frame_width)
    gaze_y = int(gaze_point[1] * frame_height)
    
    original_point = np.array([[gaze_x, gaze_y]], dtype="float32")
    transformed_point = cv2.perspectiveTransform(np.array([original_point]), homography_matrix)
    
    transformed_x, transformed_y = transformed_point[0][0]
    return int(transformed_x), int(transformed_y)


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


def interpolate_gaze(timestamps, gaze_positions, frame_time):
    """
    Linearly interpolate gaze data.
    """
    idx_before = np.searchsorted(timestamps, frame_time) - 1
    idx_after = idx_before + 1
    
    if idx_before < 0 or idx_after >= len(timestamps):
        return None
    
    t_before, t_after = timestamps[idx_before], timestamps[idx_after]
    gaze_before, gaze_after = gaze_positions[idx_before], gaze_positions[idx_after]
    
    if any(np.isnan(gaze_before)) or any(np.isnan(gaze_after)):
        return None
    
    alpha = (frame_time - t_before) / (t_after - t_before)
    return (1 - alpha) * np.array(gaze_before) + alpha * np.array(gaze_after)


############################################
# Main processing function
############################################

def process_gaze_with_perspective_correction(
    video_path,
    gaze_file_path,
    output_video_path,
    csv_output_path,
    transformation_history_path,
    output_width=1000,
    output_height=606,
    target_markers=None,
    use_preselected_parameters=False,
    use_frame_preprocessing=False,
    use_outer_points=False,
    show_video=False
):
    """
    Process video with gaze data and perspective correction.
    
    Args:
        video_path (str): Path to input video file
        gaze_file_path (str): Path to gaze data file (.gz)
        output_video_path (str): Path for output corrected video
        csv_output_path (str): Path for CSV output file
        transformation_history_path (str): Path for transformation history file
        output_width (int): Width of output video
        target_markers (list): List of marker IDs to use for perspective correction
        use_preselected_parameters (bool): Whether to use preselected ArUco parameters
        use_frame_preprocessing (bool): Whether to enhance frames before processing
        use_outer_points (bool): Whether to use outer points vs centers for correction
        show_video (bool): Whether to display video during processing
    
    Returns:
        dict: Processing results with statistics
    """
    
    # Set defaults
    if target_markers is None:
        target_markers = [13, 14, 15, 16]
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file at {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    
    # Load gaze data
    print(f"Loading gaze data from: {gaze_file_path}")
    gaze_data = load_gaze_data(gaze_file_path)
    timestamps, gaze_positions = extract_timestamps_and_gaze_positions(gaze_data)
    print(f"Loaded {len(gaze_data)} gaze samples")
    
    # Initialize CSV file
    csv_file = open(csv_output_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_index', 'frame_time', 'transformed_gaze_x', 'transformed_gaze_y'])
    
    # Initialize transformation history tracking
    transformation_history = []
    persistent_homography = None
    
    # Processing statistics
    stats = {
        'total_frames': 0,
        'frames_with_markers': 0,
        'frames_with_valid_homography': 0,
        'frames_with_gaze': 0
    }
    
    print("Processing video with gaze points and saving CSV data...")
    frame_index = 0
    
    with tqdm(total=total_frames, unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            stats['total_frames'] += 1
            frame_time = frame_index / fps
            
            # Initialize gaze coordinates as NaN
            transformed_gaze_x = np.nan
            transformed_gaze_y = np.nan
            
            # Preprocess frame if enabled
            if use_frame_preprocessing:
                preprocessed_frame = enhance_contrast_and_sharpness(frame)
            else:
                preprocessed_frame = frame
            
            # Detect ArUco markers
            gray = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            
            # Apply preselected parameters if enabled
            if use_preselected_parameters:
                parameters = apply_preselected_parameters(parameters)
            
            # Detect markers (handle different OpenCV versions)
            try:
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, _ = detector.detectMarkers(gray)
            except AttributeError:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            # Check if we can calculate a new homography matrix
            new_homography_calculated = False
            if ids is not None and len(ids) > 0:
                stats['frames_with_markers'] += 1
                target_markers_set = set(target_markers)
                detected_markers = set(ids.flatten())
                
                if target_markers_set.issubset(detected_markers):
                    # Select method to find source points
                    if use_outer_points:
                        src_points = find_outer_points(corners, ids, valid_ids=list(target_markers_set))
                    else:
                        src_points = find_and_order_average_points(corners, ids, valid_ids=list(target_markers_set))
                    
                    if src_points is not None:
                        # Define desired corners for perspective correction
                        desired_corners = np.array([
                            [0, 0],
                            [output_width - 1, 0],
                            [output_width - 1, output_height - 1],
                            [0, output_height - 1]
                        ], dtype="float32")
                        
                        # Compute homography matrix and update persistent matrix
                        H, _ = cv2.findHomography(src_points, desired_corners)
                        persistent_homography = H.copy()
                        new_homography_calculated = True
                        stats['frames_with_valid_homography'] += 1
            
            # Record the current frame's analysis results in the history
            frame_record = {
                'frame_index': frame_index,
                'frame_time': round(frame_time, 3),
                'homography_matrix': persistent_homography.copy() if persistent_homography is not None else None
            }
            transformation_history.append(frame_record)
            
            # Process video output and gaze transformation if we have a valid homography
            if persistent_homography is not None:
                corrected_frame = cv2.warpPerspective(frame, persistent_homography, (output_width, output_height))
                
                # Transform and plot gaze point
                interpolated_gaze = interpolate_gaze(timestamps, gaze_positions, frame_time)
                if interpolated_gaze is not None:
                    transformed_gaze = transform_gaze_point(interpolated_gaze, persistent_homography, frame_width, frame_height)
                    if transformed_gaze is not None:
                        tx, ty = transformed_gaze
                        if 0 <= tx < output_width and 0 <= ty < output_height:
                            transformed_gaze_x = tx
                            transformed_gaze_y = ty
                            stats['frames_with_gaze'] += 1
                            cv2.circle(corrected_frame, (tx, ty), 15, (0, 0, 255), -1)
                
                out.write(corrected_frame)
                
                if show_video:
                    cv2.imshow("Corrected Perspective with Gaze", corrected_frame)
            
            # Write CSV row for this frame
            csv_writer.writerow([frame_index, round(frame_time, 3), transformed_gaze_x, transformed_gaze_y])
            
            frame_index += 1
            pbar.update(1)
            
            if show_video and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    out.release()
    csv_file.close()
    if show_video:
        cv2.destroyAllWindows()
    
    # Save the complete transformation history to file
    np.save(transformation_history_path, transformation_history, allow_pickle=True)
    
    # Update final statistics
    stats['transformation_history_length'] = len(transformation_history)
    
    # Print results
    print("\nProcessing complete!")
    print(f"Video saved to: {output_video_path}")
    print(f"Gaze data saved to: {csv_output_path}")
    print(f"Transformation history saved to: {transformation_history_path}")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Frames with markers: {stats['frames_with_markers']}")
    print(f"Frames with valid homography: {stats['frames_with_valid_homography']}")
    print(f"Frames with gaze points: {stats['frames_with_gaze']}")
    
    return stats


