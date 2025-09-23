"""
create_final_csv_refactored.py - Modular final high-resolution gaze data CSV generation


Input files:
- gazedata.gz: Raw gaze data with timestamps
- transformation_history.npy: Frame-by-frame homography matrices

Output:
- final_gaze_data.csv: Complete gaze data with perspective corrections
"""

import numpy as np
import pandas as pd
import json
import gzip
import csv
import os
import cv2
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
        dict: A single gaze sample from the file.
    """
    logger.info(f"Streaming gaze data from: {gaze_file_path}")
    try:
        with gzip.open(gaze_file_path, 'rt') as f:
            for line in f:
                yield json.loads(line)
    except Exception as e:
        logger.error(f"Error streaming gaze data: {e}")
        # An empty generator will be returned implicitly


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


def process_gaze_stream(gaze_data_stream, transformation_history, frame_width=1920, frame_height=1080):
    """
    Process a stream of gaze samples by merging with transformation history.
    
    Args:
        gaze_data_stream (generator): A generator of raw gaze data samples, sorted by timestamp.
        transformation_history (numpy.ndarray): Transformation records, sorted by frame_time.
        frame_width (int): Original video frame width.
        frame_height (int): Original video frame height.

    Yields:
        dict: Processed gaze data records with transformations applied.
    """
    logger.info("Processing gaze stream with optimized merge logic...")

    # Pre-filter transformation history to only include valid records
    valid_transformations = [rec for rec in transformation_history if rec['homography_matrix'] is not None]
    num_valid_trans = len(valid_transformations)

    if num_valid_trans == 0:
        logger.warning("Warning: No valid transformations found in the history.")
        for gaze_sample in tqdm(gaze_data_stream, desc="Processing gaze samples (no valid transforms)"):
            yield {
                'gaze_timestamp': gaze_sample['timestamp'],
                'transformed_gaze_x': np.nan,
                'transformed_gaze_y': np.nan,
                'active_frame_index': np.nan,
                'active_frame_time': np.nan
            }
        return

    trans_idx = 0
    processed_count = 0

    # Loop through all gaze samples from the stream
    for gaze_sample in tqdm(gaze_data_stream, desc="Processing gaze samples"):
        gaze_timestamp = gaze_sample['timestamp']
        
        while (trans_idx + 1 < num_valid_trans and
               valid_transformations[trans_idx + 1]['frame_time'] <= gaze_timestamp):
            trans_idx += 1

        active_record = valid_transformations[trans_idx]

        if active_record['frame_time'] <= gaze_timestamp:
            homography_matrix = active_record['homography_matrix']
            active_frame_index = active_record['frame_index']
            active_frame_time = active_record['frame_time']
        else:
            homography_matrix = None
            active_frame_index = None
            active_frame_time = None
        
        gaze_point = gaze_sample['data'].get('gaze2d', None)
        
        transformed_x, transformed_y = transform_gaze_point(
            gaze_point, homography_matrix, frame_width, frame_height
        )
        
        yield {
            'gaze_timestamp': gaze_timestamp,
            'transformed_gaze_x': transformed_x,
            'transformed_gaze_y': transformed_y,
            'active_frame_index': active_frame_index if active_frame_index is not None else np.nan,
            'active_frame_time': active_frame_time if active_frame_time is not None else np.nan
        }
        processed_count += 1

    logger.info(f"Finished processing stream. Total gaze samples processed: {processed_count}")


def save_stream_to_csv(processed_data_stream, output_file_path):
    """
    Save a stream of processed gaze data to a CSV file and calculate stats.
    
    Args:
        processed_data_stream (generator): A generator of processed gaze records.
        output_file_path (str): Path for the output CSV file.
        
    Returns:
        dict: Statistics about the saved data, or None on failure.
    """
    logger.info(f"Saving final CSV to: {output_file_path}")
    
    total_records = 0
    valid_transformations = 0

    try:
        columns = ['gaze_timestamp', 'transformed_gaze_x', 'transformed_gaze_y',
                   'active_frame_index', 'active_frame_time']
        
        with open(output_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            for record in tqdm(processed_data_stream, desc="Writing to CSV"):
                writer.writerow(record)
                total_records += 1
                if not np.isnan(record['transformed_gaze_x']):
                    valid_transformations += 1
        
        logger.info(f"Successfully saved {total_records} records to {output_file_path}")
        
        invalid_transformations = total_records - valid_transformations
        valid_percentage = (valid_transformations / total_records) * 100 if total_records > 0 else 0
        
        stats = {
            'total_records': total_records,
            'valid_transformations': valid_transformations,
            'invalid_transformations': invalid_transformations,
            'valid_percentage': valid_percentage
        }
        
        logger.info(f"Records with valid transformations: {valid_transformations}")
        logger.info(f"Records with NaN transformations: {invalid_transformations}")
        logger.info(f"Valid transformation percentage: {stats['valid_percentage']:.1f}%")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error saving CSV file: {e}")
        return None


def create_final_gaze_csv(
    gaze_file_path,
    transformation_history_path,
    output_csv_path,
    frame_width=1920,
    frame_height=1080
):
    """
    Main function to create the final high-resolution gaze CSV file using streaming.
    
    Args:
        gaze_file_path (str): Path to the gaze data file (.gz)
        transformation_history_path (str): Path to transformation history file (.npy)
        output_csv_path (str): Path for the output CSV file
        frame_width (int): Original video frame width
        frame_height (int): Original video frame height
        
    Returns:
        dict: Processing results and statistics, or None on failure.
    """
    
    logger.info("="*60)
    logger.info("FINAL GAZE DATA CSV GENERATION (STREAMING)")
    logger.info("="*60)
    
    # Step 1: Load transformation history (still in memory, assumed to be smaller)
    transformation_history = load_transformation_history(transformation_history_path)
    if transformation_history is None:
        logger.error("Failed to load transformation history. Exiting.")
        return None

    # Step 2: Create a stream for gaze data
    gaze_data_stream = load_gaze_data_stream(gaze_file_path)

    # Step 3: Process the stream of gaze samples
    processed_data_stream = process_gaze_stream(
        gaze_data_stream, transformation_history, frame_width, frame_height
    )
    
    # Step 4: Save the processed stream to CSV
    save_stats = save_stream_to_csv(processed_data_stream, output_csv_path)
    
    if save_stats is None:
        logger.error("Failed to save CSV file. The process may have been interrupted or an error occurred.")
        return None
    
    # Compile final results
    results = {
        'success': True,
        'transformation_records': len(transformation_history),
        'output_csv_records': save_stats['total_records'],
        'valid_transformations': save_stats['valid_transformations'],
        'invalid_transformations': save_stats['invalid_transformations'],
        'valid_percentage': save_stats['valid_percentage'],
        'output_csv_path': output_csv_path,
        'frame_dimensions': (frame_width, frame_height)
    }
    
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Transformation records: {results['transformation_records']}")
    logger.info(f"Output CSV records: {results['output_csv_records']}")
    logger.info(f"Valid transformations: {results['valid_transformations']} ({results['valid_percentage']:.1f}%)")
    logger.info(f"Final CSV file: {results['output_csv_path']}")
    
    return results


