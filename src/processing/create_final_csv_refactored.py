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


def load_gaze_data(gaze_file_path):
    """
    Load and parse the raw gaze data from gazedata.gz file.
    
    Args:
        gaze_file_path (str): Path to the gaze data file
        
    Returns:
        list: List of gaze samples with timestamps and gaze2d data
    """
    logger.info(f"Loading gaze data from: {gaze_file_path}")
    
    try:
        with gzip.open(gaze_file_path, 'rt') as f:
            gaze_data = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(gaze_data)} gaze samples")
        return gaze_data
    except Exception as e:
        logger.error(f"Error loading gaze data: {e}")
        return None


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


def process_all_gaze_samples(gaze_data, transformation_history, frame_width=1920, frame_height=1080):
    """
    Process all gaze samples by merging with transformation history in a single pass.
    
    Args:
        gaze_data (list): Raw gaze data samples, sorted by timestamp.
        transformation_history (numpy.ndarray): Transformation records, sorted by frame_time.
        frame_width (int): Original video frame width.
        frame_height (int): Original video frame height.

    Returns:
        list: Processed gaze data with transformations applied.
    """
    
    logger.info("Processing all gaze samples with optimized merge logic...")
    
    processed_data = []
    
    trans_hist_idx = 0
    num_trans_hist = len(transformation_history)

    # Pre-filter transformation history to only include valid records
    valid_transformations = [rec for rec in transformation_history if rec['homography_matrix'] is not None]
    num_valid_trans = len(valid_transformations)

    if num_valid_trans == 0:
        logger.warning("Warning: No valid transformations found in the history.")
        # Process all gaze points without any transformation
        for gaze_sample in tqdm(gaze_data, desc="Processing gaze samples (no valid transforms)"):
            processed_data.append({
                'gaze_timestamp': gaze_sample['timestamp'],
                'transformed_gaze_x': np.nan,
                'transformed_gaze_y': np.nan,
                'active_frame_index': np.nan,
                'active_frame_time': np.nan
            })
        return processed_data
        

    trans_idx = 0

    # Loop through all gaze samples
    for gaze_sample in tqdm(gaze_data, desc="Processing gaze samples"):
        gaze_timestamp = gaze_sample['timestamp']
        
        # Advance transformation index until we find the correct frame
        # The correct frame is the one with the latest timestamp that is still
        # less than or equal to the gaze timestamp.
        while (trans_idx + 1 < num_valid_trans and
               valid_transformations[trans_idx + 1]['frame_time'] <= gaze_timestamp):
            trans_idx += 1

        active_record = valid_transformations[trans_idx]

        # Check if the current active_record is valid for this gaze_timestamp
        if active_record['frame_time'] <= gaze_timestamp:
            homography_matrix = active_record['homography_matrix']
            active_frame_index = active_record['frame_index']
            active_frame_time = active_record['frame_time']
        else:
            # This gaze sample is earlier than the first valid transformation
            homography_matrix = None
            active_frame_index = None
            active_frame_time = None
        
        # Extract gaze2d coordinates
        gaze_point = gaze_sample['data'].get('gaze2d', None)
        
        # Transform the gaze point
        transformed_x, transformed_y = transform_gaze_point(
            gaze_point, homography_matrix, frame_width, frame_height
        )
        
        # Create output record
        processed_record = {
            'gaze_timestamp': gaze_timestamp,
            'transformed_gaze_x': transformed_x,
            'transformed_gaze_y': transformed_y,
            'active_frame_index': active_frame_index if active_frame_index is not None else np.nan,
            'active_frame_time': active_frame_time if active_frame_time is not None else np.nan
        }
        
        processed_data.append(processed_record)

    logger.info(f"Processed {len(processed_data)} gaze samples")
    return processed_data


def save_final_csv(processed_data, output_file_path):
    """
    Save the processed gaze data to a CSV file.
    
    Args:
        processed_data (list): List of processed gaze records
        output_file_path (str): Path for the output CSV file
        
    Returns:
        dict: Statistics about the saved data
    """
    
    logger.info(f"Saving final CSV to: {output_file_path}")
    
    try:
        # Define column order
        columns = ['gaze_timestamp', 'transformed_gaze_x', 'transformed_gaze_y',
                  'active_frame_index', 'active_frame_time']
        
        # Write CSV file
        with open(output_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(processed_data)
        
        logger.info(f"Successfully saved {len(processed_data)} records to {output_file_path}")
        
        # Calculate summary statistics
        valid_transformations = sum(1 for record in processed_data
                                  if not np.isnan(record['transformed_gaze_x']))
        invalid_transformations = len(processed_data) - valid_transformations
        
        stats = {
            'total_records': len(processed_data),
            'valid_transformations': valid_transformations,
            'invalid_transformations': invalid_transformations,
            'valid_percentage': (valid_transformations / len(processed_data)) * 100 if processed_data else 0
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
    Main function to create the final high-resolution gaze CSV file.
    
    Args:
        gaze_file_path (str): Path to the gaze data file (.gz)
        transformation_history_path (str): Path to transformation history file (.npy)
        output_csv_path (str): Path for the output CSV file
        frame_width (int): Original video frame width
        frame_height (int): Original video frame height
        
    Returns:
        dict: Processing results and statistics
    """
    
    logger.info("="*60)
    logger.info("FINAL GAZE DATA CSV GENERATION")
    logger.info("="*60)
    
    # Step 1: Load input files
    gaze_data = load_gaze_data(gaze_file_path)
    if gaze_data is None:
        logger.error("Failed to load gaze data. Exiting.")
        return None
    
    transformation_history = load_transformation_history(transformation_history_path)
    if transformation_history is None:
        logger.error("Failed to load transformation history. Exiting.")
        return None
    
    # Step 2: Process all gaze samples
    processed_data = process_all_gaze_samples(
        gaze_data, transformation_history, frame_width, frame_height
    )
    
    if not processed_data:
        logger.warning("No data was processed. Exiting.")
        return None
    
    # Step 3: Save final CSV
    save_stats = save_final_csv(processed_data, output_csv_path)
    
    if save_stats is None:
        logger.error("Failed to save CSV file.")
        return None
    
    # Compile final results
    results = {
        'success': True,
        'input_gaze_samples': len(gaze_data),
        'transformation_records': len(transformation_history),
        'output_csv_records': len(processed_data),
        'valid_transformations': save_stats['valid_transformations'],
        'invalid_transformations': save_stats['invalid_transformations'],
        'valid_percentage': save_stats['valid_percentage'],
        'output_csv_path': output_csv_path,
        'frame_dimensions': (frame_width, frame_height)
    }
    
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Input gaze samples: {results['input_gaze_samples']}")
    logger.info(f"Transformation records: {results['transformation_records']}")
    logger.info(f"Output CSV records: {results['output_csv_records']}")
    logger.info(f"Valid transformations: {results['valid_transformations']} ({results['valid_percentage']:.1f}%)")
    logger.info(f"Final CSV file: {results['output_csv_path']}")
    
    return results


