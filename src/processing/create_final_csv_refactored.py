"""
create_final_csv_refactored.py - Enhanced final high-resolution gaze data CSV generation

Physics Update: Now includes angular velocity (degrees), pupil diameter, frame luminance,
and IMU gyroscope data for whole-session analysis.

Input files:
- gazedata.gz: Raw gaze data with timestamps, 3D gaze direction, pupil diameter
- imudata.gz: IMU data with gyroscope readings
- transformation_history.npy: Frame-by-frame homography matrices
- scenevideo.mp4: Video for luminance extraction

Output:
- final_gaze_data.csv: Complete gaze data with physics-based metrics
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
    Load and parse the raw gaze data from gazedata.gz file as a stream.

    Args:
        gaze_file_path (str): Path to the gaze data file

    Yields:
        dict: A single gaze sample from the file with enhanced data extraction.
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
    Load and parse IMU data from imudata.gz file.

    Args:
        imu_file_path (str): Path to the IMU data file

    Returns:
        list: List of IMU records with timestamps and gyroscope data
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
    Extract mean luminance (grayscale intensity) for each frame.

    Args:
        video_path (str): Path to the video file
        sample_rate (int): Process every Nth frame (1 = all frames)

    Returns:
        dict: Mapping of frame_index to luminance value (0-255)
    """
    logger.info(f"Extracting frame luminances from: {video_path}")
    luminances = {}

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return luminances

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Video: {total_frames} frames at {fps:.2f} FPS")

        frame_idx = 0
        with tqdm(total=total_frames, desc="Extracting luminance") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    # Convert to grayscale and calculate mean intensity
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    luminances[frame_idx] = float(np.mean(gray))

                frame_idx += 1
                pbar.update(1)

        cap.release()
        logger.info(f"Extracted luminance for {len(luminances)} frames")
        return luminances

    except Exception as e:
        logger.error(f"Error extracting luminances: {e}")
        return luminances


def calculate_angular_velocity(gaze_dir_current, gaze_dir_previous, delta_t):
    """
    Calculate angular velocity between two 3D gaze direction vectors.

    Uses the formula: theta = arccos(v_t . v_{t-1})
    Angular velocity = theta / delta_t (converted to degrees/second)

    Args:
        gaze_dir_current: Current 3D unit vector [x, y, z]
        gaze_dir_previous: Previous 3D unit vector [x, y, z]
        delta_t: Time difference in seconds

    Returns:
        float: Angular velocity in degrees/second, or NaN if invalid
    """
    if gaze_dir_current is None or gaze_dir_previous is None:
        return np.nan

    if delta_t <= 0 or delta_t > 1.0:  # Sanity check on time delta
        return np.nan

    try:
        v_curr = np.array(gaze_dir_current)
        v_prev = np.array(gaze_dir_previous)

        # Normalize vectors (should already be unit vectors, but ensure)
        norm_curr = np.linalg.norm(v_curr)
        norm_prev = np.linalg.norm(v_prev)

        if norm_curr < 1e-6 or norm_prev < 1e-6:
            return np.nan

        v_curr = v_curr / norm_curr
        v_prev = v_prev / norm_prev

        # Calculate dot product and clamp to valid range for arccos
        dot_product = np.dot(v_curr, v_prev)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate angular change in radians, then convert to degrees
        theta_radians = np.arccos(dot_product)
        theta_degrees = np.degrees(theta_radians)

        # Calculate angular velocity (degrees/second)
        angular_velocity = theta_degrees / delta_t

        return angular_velocity

    except Exception:
        return np.nan


def extract_gaze_direction(gaze_sample):
    """
    Extract averaged gaze direction from left and right eye data.

    Args:
        gaze_sample: Raw gaze sample dictionary

    Returns:
        tuple: (averaged_gaze_direction, left_pupil, right_pupil)
    """
    data = gaze_sample.get('data', {})

    # Extract left eye data
    left_eye = data.get('eyeleft', {})
    left_dir = left_eye.get('gazedirection', None)
    left_pupil = left_eye.get('pupildiameter', None)

    # Extract right eye data
    right_eye = data.get('eyeright', {})
    right_dir = right_eye.get('gazedirection', None)
    right_pupil = right_eye.get('pupildiameter', None)

    # Average the gaze directions if both are available
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
    Find the nearest IMU record for a given gaze timestamp.

    Args:
        gaze_timestamp: Target timestamp
        imu_records: List of IMU records
        imu_index_hint: Starting index hint for search optimization

    Returns:
        tuple: (gyro_x, gyro_y, gyro_z, new_index_hint)
    """
    if not imu_records:
        return None, None, None, 0

    # Start search from hint index
    idx = imu_index_hint
    n = len(imu_records)

    # Move forward until we pass the target timestamp
    while idx + 1 < n and imu_records[idx + 1]['timestamp'] <= gaze_timestamp:
        idx += 1

    # Return the nearest IMU record
    if idx < n:
        record = imu_records[idx]
        return record['gyro_x'], record['gyro_y'], record['gyro_z'], idx

    return None, None, None, idx


def load_transformation_history(history_file_path):
    """
    Load the transformation history from the .npy file.

    Args:
        history_file_path (str): Path to the transformation history file

    Returns:
        Array of transformation history records
    """
    logger.info(f"Loading transformation history from: {history_file_path}")

    try:
        transformation_history = np.load(history_file_path, allow_pickle=True)
        logger.info(f"Loaded {len(transformation_history)} transformation records")

        # Count valid transformations
        valid_transformations = sum(1 for record in transformation_history
                                  if record['homography_matrix'] is not None)
        logger.info(f"Found {valid_transformations} frames with valid homography matrices")

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
    Process gaze samples with physics-based enhancements.

    Adds: angular velocity, pupil diameter, frame luminance, gyroscope data.

    Args:
        gaze_data_stream: Generator of raw gaze data samples
        transformation_history: Transformation records
        imu_records: List of IMU records
        luminance_lookup: Dict mapping frame_index to luminance
        fps: Video frames per second
        frame_width: Original video frame width
        frame_height: Original video frame height

    Yields:
        dict: Enhanced processed gaze records
    """
    logger.info("Processing gaze stream with physics-based enhancements...")

    # Pre-filter transformation history
    valid_transformations = [rec for rec in transformation_history if rec['homography_matrix'] is not None]
    num_valid_trans = len(valid_transformations)

    # Even if no valid transformations, we still extract physics-based data
    no_valid_transforms = (num_valid_trans == 0)
    if no_valid_transforms:
        logger.warning("No valid transformations found. Will still extract physics-based data.")

    trans_idx = 0
    imu_idx = 0
    prev_gaze_dir = None
    prev_timestamp = None
    processed_count = 0

    for gaze_sample in tqdm(gaze_data_stream, desc="Processing gaze samples"):
        gaze_timestamp = gaze_sample['timestamp']

        # Handle transformation lookup
        homography_matrix = None
        active_frame_index = None
        active_frame_time = None
        transformed_x, transformed_y = np.nan, np.nan

        if not no_valid_transforms:
            # Find active transformation
            while (trans_idx + 1 < num_valid_trans and
                   valid_transformations[trans_idx + 1]['frame_time'] <= gaze_timestamp):
                trans_idx += 1

            active_record = valid_transformations[trans_idx]

            if active_record['frame_time'] <= gaze_timestamp:
                homography_matrix = active_record['homography_matrix']
                active_frame_index = active_record['frame_index']
                active_frame_time = active_record['frame_time']

            # Transform gaze point
            gaze_point = gaze_sample['data'].get('gaze2d', None)
            transformed_x, transformed_y = transform_gaze_point(
                gaze_point, homography_matrix, frame_width, frame_height
            )
        else:
            # Estimate frame index from timestamp when no valid transforms
            estimated_frame = int(gaze_timestamp * fps) if fps > 0 else None
            active_frame_index = estimated_frame
            active_frame_time = gaze_timestamp

        # Extract gaze direction and pupil data
        gaze_dir, left_pupil, right_pupil = extract_gaze_direction(gaze_sample)

        # Calculate angular velocity
        angular_velocity = np.nan
        if prev_gaze_dir is not None and prev_timestamp is not None:
            delta_t = gaze_timestamp - prev_timestamp
            angular_velocity = calculate_angular_velocity(gaze_dir, prev_gaze_dir, delta_t)

        # Calculate average pupil diameter
        if left_pupil is not None and right_pupil is not None:
            avg_pupil = (left_pupil + right_pupil) / 2
        elif left_pupil is not None:
            avg_pupil = left_pupil
        elif right_pupil is not None:
            avg_pupil = right_pupil
        else:
            avg_pupil = np.nan

        # Get frame luminance
        frame_luminance = np.nan
        if active_frame_index is not None:
            frame_luminance = luminance_lookup.get(active_frame_index, np.nan)

        # Get synchronized IMU data
        gyro_x, gyro_y, gyro_z, imu_idx = sync_imu_to_timestamp(
            gaze_timestamp, imu_records, imu_idx
        )

        # Update state for next iteration
        prev_gaze_dir = gaze_dir
        prev_timestamp = gaze_timestamp

        yield {
            'gaze_timestamp': gaze_timestamp,
            'transformed_gaze_x': transformed_x,
            'transformed_gaze_y': transformed_y,
            'active_frame_index': active_frame_index if active_frame_index is not None else np.nan,
            'active_frame_time': active_frame_time if active_frame_time is not None else np.nan,
            'angular_velocity_deg_s': angular_velocity,
            'pupil_diameter_left': left_pupil if left_pupil is not None else np.nan,
            'pupil_diameter_right': right_pupil if right_pupil is not None else np.nan,
            'pupil_diameter_avg': avg_pupil if not np.isnan(avg_pupil) else np.nan,
            'frame_luminance': frame_luminance,
            'head_gyro_x': gyro_x if gyro_x is not None else np.nan,
            'head_gyro_y': gyro_y if gyro_y is not None else np.nan,
            'head_gyro_z': gyro_z if gyro_z is not None else np.nan
        }
        processed_count += 1

    logger.info(f"Finished processing stream. Total: {processed_count}")


def _create_empty_record(timestamp):
    """Create an empty record with NaN values."""
    return {
        'gaze_timestamp': timestamp,
        'transformed_gaze_x': np.nan,
        'transformed_gaze_y': np.nan,
        'active_frame_index': np.nan,
        'active_frame_time': np.nan,
        'angular_velocity_deg_s': np.nan,
        'pupil_diameter_left': np.nan,
        'pupil_diameter_right': np.nan,
        'pupil_diameter_avg': np.nan,
        'frame_luminance': np.nan,
        'head_gyro_x': np.nan,
        'head_gyro_y': np.nan,
        'head_gyro_z': np.nan
    }


def save_stream_to_csv(processed_data_stream, output_file_path):
    """
    Save processed gaze data stream to CSV and calculate statistics.

    Args:
        processed_data_stream: Generator of processed gaze records
        output_file_path: Path for output CSV file

    Returns:
        dict: Statistics about the saved data
    """
    logger.info(f"Saving enhanced CSV to: {output_file_path}")

    try:
        df = pd.DataFrame(list(processed_data_stream))

        if df.empty:
            logger.warning("No data to save.")
            columns = [
                'gaze_timestamp', 'transformed_gaze_x', 'transformed_gaze_y',
                'active_frame_index', 'active_frame_time', 'angular_velocity_deg_s',
                'pupil_diameter_left', 'pupil_diameter_right', 'pupil_diameter_avg',
                'frame_luminance', 'head_gyro_x', 'head_gyro_y', 'head_gyro_z'
            ]
            pd.DataFrame(columns=columns).to_csv(output_file_path, index=False)
            return {
                'total_records': 0,
                'valid_transformations': 0,
                'invalid_transformations': 0,
                'valid_percentage': 0,
                'valid_angular_velocity': 0,
                'valid_pupil': 0,
                'valid_imu': 0
            }

        df.to_csv(output_file_path, index=False)

        # Calculate statistics
        total_records = len(df)
        valid_transformations = df['transformed_gaze_x'].notna().sum()
        valid_angular = df['angular_velocity_deg_s'].notna().sum()
        valid_pupil = df['pupil_diameter_avg'].notna().sum()
        valid_imu = df['head_gyro_x'].notna().sum()
        valid_luminance = df['frame_luminance'].notna().sum()

        stats = {
            'total_records': total_records,
            'valid_transformations': int(valid_transformations),
            'invalid_transformations': int(total_records - valid_transformations),
            'valid_percentage': (valid_transformations / total_records) * 100 if total_records > 0 else 0,
            'valid_angular_velocity': int(valid_angular),
            'valid_angular_velocity_pct': (valid_angular / total_records) * 100 if total_records > 0 else 0,
            'valid_pupil': int(valid_pupil),
            'valid_pupil_pct': (valid_pupil / total_records) * 100 if total_records > 0 else 0,
            'valid_imu': int(valid_imu),
            'valid_imu_pct': (valid_imu / total_records) * 100 if total_records > 0 else 0,
            'valid_luminance': int(valid_luminance),
            'valid_luminance_pct': (valid_luminance / total_records) * 100 if total_records > 0 else 0
        }

        logger.info(f"Saved {total_records} records to {output_file_path}")
        logger.info(f"  Valid transformations: {stats['valid_transformations']} ({stats['valid_percentage']:.1f}%)")
        logger.info(f"  Valid angular velocity: {stats['valid_angular_velocity']} ({stats['valid_angular_velocity_pct']:.1f}%)")
        logger.info(f"  Valid pupil data: {stats['valid_pupil']} ({stats['valid_pupil_pct']:.1f}%)")
        logger.info(f"  Valid IMU data: {stats['valid_imu']} ({stats['valid_imu_pct']:.1f}%)")
        logger.info(f"  Valid luminance: {stats['valid_luminance']} ({stats['valid_luminance_pct']:.1f}%)")

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
    Create the enhanced final high-resolution gaze CSV with physics-based metrics.

    Args:
        gaze_file_path: Path to gazedata.gz
        transformation_history_path: Path to transformation_history.npy
        output_csv_path: Path for output CSV
        frame_width: Video frame width
        frame_height: Video frame height
        video_path: Path to scenevideo.mp4 (for luminance extraction)
        imu_file_path: Path to imudata.gz

    Returns:
        dict: Processing results and statistics
    """
    logger.info("=" * 60)
    logger.info("ENHANCED FINAL GAZE DATA CSV GENERATION")
    logger.info("Physics Update: Angular velocity, pupil, luminance, IMU")
    logger.info("=" * 60)

    # Auto-detect file paths if not provided
    base_dir = os.path.dirname(gaze_file_path)
    if video_path is None:
        video_path = os.path.join(base_dir, 'scenevideo.mp4')
    if imu_file_path is None:
        imu_file_path = os.path.join(base_dir, 'imudata.gz')

    # Step 1: Load transformation history
    transformation_history = load_transformation_history(transformation_history_path)
    if transformation_history is None:
        logger.error("Failed to load transformation history.")
        return None

    # Step 2: Load IMU data
    imu_records = []
    if os.path.exists(imu_file_path):
        imu_records = load_imu_data(imu_file_path)
    else:
        logger.warning(f"IMU file not found: {imu_file_path}")

    # Step 3: Extract frame luminances
    luminance_lookup = {}
    fps = 30.0  # Default FPS
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        luminance_lookup = extract_frame_luminances(video_path)
    else:
        logger.warning(f"Video file not found: {video_path}")

    # Step 4: Create gaze data stream
    gaze_data_stream = load_gaze_data_stream(gaze_file_path)

    # Step 5: Process the enhanced stream
    processed_data_stream = process_gaze_stream_enhanced(
        gaze_data_stream,
        transformation_history,
        imu_records,
        luminance_lookup,
        fps,
        frame_width,
        frame_height
    )

    # Step 6: Save to CSV
    save_stats = save_stream_to_csv(processed_data_stream, output_csv_path)

    if save_stats is None:
        logger.error("Failed to save CSV.")
        return None

    results = {
        'success': True,
        'transformation_records': len(transformation_history),
        'imu_records': len(imu_records),
        'luminance_frames': len(luminance_lookup),
        'output_csv_records': save_stats['total_records'],
        'valid_transformations': save_stats['valid_transformations'],
        'invalid_transformations': save_stats['invalid_transformations'],
        'valid_percentage': save_stats['valid_percentage'],
        'valid_angular_velocity': save_stats.get('valid_angular_velocity', 0),
        'valid_pupil': save_stats.get('valid_pupil', 0),
        'valid_imu': save_stats.get('valid_imu', 0),
        'output_csv_path': output_csv_path,
        'frame_dimensions': (frame_width, frame_height)
    }

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output CSV: {results['output_csv_path']}")
    logger.info(f"Total records: {results['output_csv_records']}")
    logger.info(f"Valid transformations: {results['valid_transformations']} ({results['valid_percentage']:.1f}%)")
    logger.info(f"Valid angular velocity: {results['valid_angular_velocity']}")
    logger.info(f"Valid pupil data: {results['valid_pupil']}")
    logger.info(f"Valid IMU data: {results['valid_imu']}")

    return results
