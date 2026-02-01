"""
create_final_csv_refactored.py - Enhanced final high-resolution gaze data CSV generation

This script creates the "Golden Source" dataset for analysis.
It combines data from multiple sources:
1.  **Gaze Data**: Raw eye tracking data (JSON).
2.  **Transformation History**: The perspective correction matrices calculated in Step 1.
3.  **IMU Data**: Gyroscope/accelerometer data for head movement analysis.
4.  **Scene Video**: To extract frame brightness (luminance) for pupil analysis.

Output:
    A single CSV file where every row is a gaze sample, enriched with:
    - Stabilized (x, y) coordinates
    - Physics-based angular velocity (deg/s)
    - Pupil diameter (mm)
    - Head rotation speed (deg/s)
    - Environmental brightness (luminance)
"""

import numpy as np
import pandas as pd
import json
import gzip
import cv2
import os
from tqdm import tqdm
from .utils import transform_gaze_point
from ..logging_config import get_logger

logger = get_logger(__name__)


def load_gaze_data_stream(gaze_file_path):
    """
    Load raw gaze data one line at a time (generator).
    This avoids loading massive files entirely into RAM.
    """
    logger.info(f"Streaming gaze data from: {gaze_file_path}")
    try:
        with gzip.open(gaze_file_path, 'rt') as f:
            for line in f:
                yield json.loads(line)
    except Exception as e:
        logger.error(f"Error streaming gaze data: {e}")


def load_imu_data(imu_file_path):
    """
    Load Inertial Measurement Unit (IMU) data.
    This tells us how fast the user's head is moving.
    """
    logger.info(f"Loading IMU data from: {imu_file_path}")
    imu_records = []

    try:
        with gzip.open(imu_file_path, 'rt') as f:
            for line in f:
                record = json.loads(line)
                if record.get('type') == 'imu' and 'data' in record:
                    gyro = record['data'].get('gyroscope', [None, None, None])
                    imu_records.append({
                        'timestamp': record.get('timestamp', 0),
                        'gyro_x': gyro[0] if len(gyro) > 0 else None,
                        'gyro_y': gyro[1] if len(gyro) > 1 else None,
                        'gyro_z': gyro[2] if len(gyro) > 2 else None
                    })

        logger.info(f"Loaded {len(imu_records)} IMU records")
        return imu_records
    except FileNotFoundError:
        logger.warning(f"IMU file not found: {imu_file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading IMU data: {e}")
        return []


def extract_frame_luminances(video_path, sample_rate=1):
    """
    Calculate the average brightness of each video frame.

    Why? The pupil constricts in bright light. To measure cognitive load
    (which dilates the pupil), we must first account for the light reflex.
    """
    logger.info(f"Extracting frame luminances from: {video_path}")
    luminances = {}

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return luminances

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        # tqdm shows a progress bar
        with tqdm(total=total_frames, desc="Extracting luminance") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    # Convert to grayscale (0-255 intensity)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Average the intensity of all pixels
                    luminances[frame_idx] = float(np.mean(gray))

                frame_idx += 1
                pbar.update(1)

        cap.release()
        return luminances

    except Exception as e:
        logger.error(f"Error extracting luminances: {e}")
        return luminances


def calculate_angular_velocity(gaze_dir_current, gaze_dir_previous, delta_t):
    """
    Calculate how fast the eye rotated between two samples.

    Formula: Velocity = Angle / Time
    Angle = arccos(dot_product(vector1, vector2))
    """
    if gaze_dir_current is None or gaze_dir_previous is None:
        return np.nan

    if delta_t <= 0 or delta_t > 1.0:  # Sanity check: prevent division by zero or huge jumps
        return np.nan

    try:
        v_curr = np.array(gaze_dir_current)
        v_prev = np.array(gaze_dir_previous)

        # Normalize vectors to length 1 (Unit Vectors)
        norm_curr = np.linalg.norm(v_curr)
        norm_prev = np.linalg.norm(v_prev)

        if norm_curr < 1e-6 or norm_prev < 1e-6:
            return np.nan

        v_curr = v_curr / norm_curr
        v_prev = v_prev / norm_prev

        # Calculate angle using Dot Product
        # A . B = |A||B|cos(theta) -> theta = arccos(A . B)
        dot_product = np.dot(v_curr, v_prev)
        dot_product = np.clip(dot_product, -1.0, 1.0) # Ensure within valid range [-1, 1]

        theta_radians = np.arccos(dot_product)
        theta_degrees = np.degrees(theta_radians)

        angular_velocity = theta_degrees / delta_t

        return angular_velocity

    except Exception:
        return np.nan


def extract_gaze_direction(gaze_sample):
    """
    Get the 3D gaze vector. If both eyes are tracked, average them.
    """
    data = gaze_sample.get('data', {})
    left_eye = data.get('eyeleft', {})
    right_eye = data.get('eyeright', {})

    left_dir = left_eye.get('gazedirection', None)
    right_dir = right_eye.get('gazedirection', None)

    left_pupil = left_eye.get('pupildiameter', None)
    right_pupil = right_eye.get('pupildiameter', None)

    if left_dir is not None and right_dir is not None:
        avg_dir = [
            (left_dir[0] + right_dir[0]) / 2,
            (left_dir[1] + right_dir[1]) / 2,
            (left_dir[2] + right_dir[2]) / 2
        ]
    elif left_dir is not None:
        avg_dir = left_dir
    elif right_dir is not None:
        avg_dir = right_dir
    else:
        avg_dir = None

    return avg_dir, left_pupil, right_pupil


def sync_imu_to_timestamp(gaze_timestamp, imu_records, imu_index_hint=0):
    """
    Match a gaze sample timestamp to the nearest IMU sample.
    Since IMU data might be recorded at a different rate, we search for the closest match.
    """
    if not imu_records:
        return None, None, None, 0

    idx = imu_index_hint
    n = len(imu_records)

    # Fast-forward until we pass the timestamp
    while idx + 1 < n and imu_records[idx + 1]['timestamp'] <= gaze_timestamp:
        idx += 1

    if idx < n:
        record = imu_records[idx]
        return record['gyro_x'], record['gyro_y'], record['gyro_z'], idx

    return None, None, None, idx


def load_transformation_history(history_file_path):
    """
    Load the Homography Matrices saved in Step 1.
    """
    logger.info(f"Loading transformation history from: {history_file_path}")
    try:
        transformation_history = np.load(history_file_path, allow_pickle=True)
        return transformation_history
    except Exception as e:
        logger.error(f"Error loading transformation history: {e}")
        return None


def process_gaze_stream_enhanced(
    gaze_data_stream,
    transformation_history,
    imu_records,
    luminance_lookup,
    fps,
    frame_width=1920,
    frame_height=1080
):
    """
    The main processing loop. Iterates through every gaze sample and:
    1. Finds the corresponding video frame.
    2. Applies the homography (if valid) to get stabilized coordinates.
    3. Calculates physics metrics (velocity, pupil).
    4. Syncs with IMU and Luminance data.
    """
    logger.info("Processing gaze stream with physics-based enhancements...")

    # Filter out frames where no markers were found (invalid homography)
    valid_transformations = [rec for rec in transformation_history if rec['homography_matrix'] is not None]
    num_valid_trans = len(valid_transformations)
    no_valid_transforms = (num_valid_trans == 0)

    trans_idx = 0
    imu_idx = 0
    prev_gaze_dir = None
    prev_timestamp = None

    for gaze_sample in tqdm(gaze_data_stream, desc="Processing gaze samples"):
        gaze_timestamp = gaze_sample['timestamp']

        # --- 1. Coordinate Transformation ---
        homography_matrix = None
        active_frame_index = None
        active_frame_time = None
        transformed_x, transformed_y = np.nan, np.nan

        if not no_valid_transforms:
            # Find the transformation matrix for this specific time
            while (trans_idx + 1 < num_valid_trans and
                   valid_transformations[trans_idx + 1]['frame_time'] <= gaze_timestamp):
                trans_idx += 1

            active_record = valid_transformations[trans_idx]

            # Only apply if the timestamp is reasonably close (within the video duration)
            if active_record['frame_time'] <= gaze_timestamp:
                homography_matrix = active_record['homography_matrix']
                active_frame_index = active_record['frame_index']
                active_frame_time = active_record['frame_time']

            # Apply mathematical transformation (Perspective Warp)
            gaze_point = gaze_sample['data'].get('gaze2d', None)
            transformed_x, transformed_y = transform_gaze_point(
                gaze_point, homography_matrix, frame_width, frame_height
            )
        else:
            # Fallback if no perspective correction is possible
            estimated_frame = int(gaze_timestamp * fps) if fps > 0 else None
            active_frame_index = estimated_frame

        # --- 2. Physics Metrics ---
        gaze_dir, left_pupil, right_pupil = extract_gaze_direction(gaze_sample)

        # Angular Velocity calculation
        angular_velocity = np.nan
        if prev_gaze_dir is not None and prev_timestamp is not None:
            delta_t = gaze_timestamp - prev_timestamp
            angular_velocity = calculate_angular_velocity(gaze_dir, prev_gaze_dir, delta_t)

        # Pupil Diameter
        if left_pupil is not None and right_pupil is not None:
            avg_pupil = (left_pupil + right_pupil) / 2
        elif left_pupil is not None:
            avg_pupil = left_pupil
        elif right_pupil is not None:
            avg_pupil = right_pupil
        else:
            avg_pupil = np.nan

        # --- 3. Environmental Data ---
        frame_luminance = np.nan
        if active_frame_index is not None:
            frame_luminance = luminance_lookup.get(active_frame_index, np.nan)

        # IMU (Head Movement)
        gyro_x, gyro_y, gyro_z, imu_idx = sync_imu_to_timestamp(
            gaze_timestamp, imu_records, imu_idx
        )

        prev_gaze_dir = gaze_dir
        prev_timestamp = gaze_timestamp

        # Yield the fully enriched data row
        yield {
            'gaze_timestamp': gaze_timestamp,
            'transformed_gaze_x': transformed_x,
            'transformed_gaze_y': transformed_y,
            'active_frame_index': active_frame_index if active_frame_index is not None else np.nan,
            'active_frame_time': active_frame_time if active_frame_time is not None else np.nan,
            'angular_velocity_deg_s': angular_velocity,
            'gaze_direction_x': gaze_dir[0] if gaze_dir is not None else np.nan,
            'gaze_direction_y': gaze_dir[1] if gaze_dir is not None else np.nan,
            'gaze_direction_z': gaze_dir[2] if gaze_dir is not None else np.nan,
            'pupil_diameter_avg': avg_pupil if not np.isnan(avg_pupil) else np.nan,
            'frame_luminance': frame_luminance,
            'head_gyro_x': gyro_x if gyro_x is not None else np.nan,
            'head_gyro_y': gyro_y if gyro_y is not None else np.nan,
            'head_gyro_z': gyro_z if gyro_z is not None else np.nan
        }


def save_stream_to_csv(processed_data_stream, output_file_path):
    """
    Consume the data stream and write it to a CSV file.
    """
    logger.info(f"Saving enhanced CSV to: {output_file_path}")

    try:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(list(processed_data_stream))

        if df.empty:
            logger.warning("No data to save.")
            return None

        # Write to disk
        df.to_csv(output_file_path, index=False)

        # Calculate basic quality statistics
        total_records = len(df)
        valid_transformations = df['transformed_gaze_x'].notna().sum()

        stats = {
            'total_records': total_records,
            'valid_transformations': int(valid_transformations),
            'invalid_transformations': int(total_records - valid_transformations),
            'valid_percentage': (valid_transformations / total_records) * 100 if total_records > 0 else 0,
        }
        return stats

    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        return None


def create_final_gaze_csv(
    gaze_file_path,
    transformation_history_path,
    output_csv_path,
    frame_width=1920,
    frame_height=1080,
    video_path=None,
    imu_file_path=None
):
    """
    Main entry point for CSV creation.
    """
    # Auto-detect file paths if not provided
    base_dir = os.path.dirname(gaze_file_path)
    if video_path is None:
        video_path = os.path.join(base_dir, 'scenevideo.mp4')
    if imu_file_path is None:
        imu_file_path = os.path.join(base_dir, 'imudata.gz')

    # Load inputs
    transformation_history = load_transformation_history(transformation_history_path)
    if transformation_history is None: return None

    imu_records = []
    if os.path.exists(imu_file_path):
        imu_records = load_imu_data(imu_file_path)

    luminance_lookup = {}
    fps = 30.0
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        luminance_lookup = extract_frame_luminances(video_path)

    # Process
    gaze_data_stream = load_gaze_data_stream(gaze_file_path)

    processed_data_stream = process_gaze_stream_enhanced(
        gaze_data_stream,
        transformation_history,
        imu_records,
        luminance_lookup,
        fps,
        frame_width,
        frame_height
    )

    save_stats = save_stream_to_csv(processed_data_stream, output_csv_path)

    if save_stats:
        return {'success': True, **save_stats}
    return {'success': False}
