"""
create_final_csv.py - Generate final high-resolution gaze data CSV

This script processes raw gaze data using the transformation history to produce
perspective-corrected gaze coordinates for every single gaze sample.

Input files:
- gazedata.gz: Raw gaze data with timestamps
- transformation_history.npy: Frame-by-frame homography matrices

"""

import numpy as np
import pandas as pd
import json
import gzip
import csv
import os
import cv2
from tqdm import tqdm


def load_gaze_data(gaze_file_path):
    """
    Load and parse the raw gaze data from gazedata.gz file.
    
    Returns:
        list: List of gaze samples with timestamps and gaze2d data
    """
    print(f"Loading gaze data from: {gaze_file_path}")
    
    try:
        with gzip.open(gaze_file_path, 'rt') as f:
            gaze_data = [json.loads(line) for line in f]
        print(f"Loaded {len(gaze_data)} gaze samples")
        return gaze_data
    except Exception as e:
        print(f"Error loading gaze data: {e}")
        return None


def load_transformation_history(history_file_path):
    """
    Load the transformation history from the .npy file.
    
    Returns:
        numpy.ndarray: Array of transformation history records
    """
    print(f"Loading transformation history from: {history_file_path}")
    
    try:
        transformation_history = np.load(history_file_path, allow_pickle=True)
        print(f"Loaded {len(transformation_history)} transformation records")
        
        # Count valid transformations
        valid_transformations = sum(1 for record in transformation_history
                                  if record['homography_matrix'] is not None)
        print(f"Found {valid_transformations} frames with valid homography matrices")
        
        return transformation_history
    except Exception as e:
        print(f"Error loading transformation history: {e}")
        return None


def find_active_transformation(gaze_timestamp, transformation_history):
    """
    Find the active transformation matrix for a given gaze timestamp.
    
    Args:
        gaze_timestamp (float): Timestamp of the gaze sample
        transformation_history (numpy.ndarray): Array of transformation records
    
    Returns:
        (homography_matrix, frame_index, frame_time) or (None, None, None)
    """
    
    # Find the most recent frame_time that is <= gaze_timestamp
    active_record = None
    
    for record in transformation_history:
        frame_time = record['frame_time']
        
        # Check if this frame time is before or at the gaze timestamp
        if frame_time <= gaze_timestamp:
            active_record = record
        else:
            # Since transformation_history is ordered by frame_time,
            # we can break once we find a frame_time > gaze_timestamp
            break
    
    if active_record is not None:
        return (active_record['homography_matrix'],
                active_record['frame_index'],
                active_record['frame_time'])
    else:
        # No frame found before this timestamp (very early gaze data)
        return (None, None, None)


def transform_gaze_point(gaze_point, homography_matrix, frame_width=1920, frame_height=1080):
    """
    Apply homography transformation to a gaze point.
    
    Args:
        gaze_point (list): Normalized gaze coordinates [x, y] (0-1 range)
        homography_matrix (numpy.ndarray): 3x3 homography matrix
        frame_width (int): Original frame width in pixels
        frame_height (int): Original frame height in pixels
    
    Returns:
        (transformed_x, transformed_y) or (NaN, NaN) if invalid
    """
    
    # Check for invalid gaze point
    if gaze_point is None or len(gaze_point) != 2:
        return (np.nan, np.nan)
    
    if any(np.isnan(gaze_point)) or homography_matrix is None:
        return (np.nan, np.nan)
    
    try:
        # Convert normalized gaze2d to pixel coordinates
        gaze_x = gaze_point[0] * frame_width
        gaze_y = gaze_point[1] * frame_height
        
        # Apply homography transformation
        original_point = np.array([[gaze_x, gaze_y]], dtype="float32")
        transformed_point = cv2.perspectiveTransform(np.array([original_point]), homography_matrix)
        
        # Return transformed coordinates
        transformed_x, transformed_y = transformed_point[0][0]
        return (float(transformed_x), float(transformed_y))
        
    except Exception as e:
        print(f"Warning: Error transforming gaze point {gaze_point}: {e}")
        return (np.nan, np.nan)


def process_all_gaze_samples(gaze_data, transformation_history, frame_width=1920, frame_height=1080):
    """
    Process all gaze samples and apply perspective corrections.
    
    Args:
        gaze_data (list): Raw gaze data samples
        transformation_history (numpy.ndarray): Transformation history records
        frame_width (int): Original video frame width
        frame_height (int): Original video frame height
    
    Returns:
        list: Processed gaze data with transformations applied
    """
    
    print("Processing all gaze samples...")
    
    processed_data = []
    
    # Process each gaze sample
    for i, gaze_sample in enumerate(tqdm(gaze_data, desc="Processing gaze samples")):
        
        # Extract timestamp and gaze coordinates
        gaze_timestamp = gaze_sample['timestamp']
        
        # Extract gaze2d coordinates if available
        if 'gaze2d' in gaze_sample['data']:
            gaze_point = gaze_sample['data']['gaze2d']
        else:
            gaze_point = None
        
        # Find the active transformation for this timestamp
        homography_matrix, active_frame_index, active_frame_time = find_active_transformation(
            gaze_timestamp, transformation_history
        )
        
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
    
    print(f"Processed {len(processed_data)} gaze samples")
    return processed_data


def save_final_csv(processed_data, output_file_path):
    """
    Save the processed gaze data to a CSV file.
    
    Args:
        processed_data (list): List of processed gaze records
        output_file_path (str): Path for the output CSV file
    """
    
    print(f"Saving final CSV to: {output_file_path}")
    
    try:
        # Define column order
        columns = ['gaze_timestamp', 'transformed_gaze_x', 'transformed_gaze_y',
                  'active_frame_index', 'active_frame_time']
        
        # Write CSV file
        with open(output_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(processed_data)
        
        print(f"Successfully saved {len(processed_data)} records to {output_file_path}")
        
        # Print summary statistics
        valid_transformations = sum(1 for record in processed_data
                                  if not np.isnan(record['transformed_gaze_x']))
        print(f"Records with valid transformations: {valid_transformations}")
        print(f"Records with NaN transformations: {len(processed_data) - valid_transformations}")
        
    except Exception as e:
        print(f"Error saving CSV file: {e}")


def main():
    """
    Main function to orchestrate the final CSV generation process.
    """
    
    print("="*60)
    print("FINAL GAZE DATA CSV GENERATION")
    print("="*60)
    
    # File paths (adjust these to match your setup)
    gaze_file_path = 'gazedata.gz'
    transformation_history_path = 'transformation_history.npy'
    output_csv_path = 'final_gaze_data.csv'
    
    # Video properties (should match the original video used to generate transformations)
    frame_width = 1920  # Adjust if your video has different dimensions
    frame_height = 1080
    
    # Step 1: Load input files
    gaze_data = load_gaze_data(gaze_file_path)
    if gaze_data is None:
        print("Failed to load gaze data. Exiting.")
        return False
    
    transformation_history = load_transformation_history(transformation_history_path)
    if transformation_history is None:
        print("Failed to load transformation history. Exiting.")
        return False
    
    # Step 2: Process all gaze samples
    processed_data = process_all_gaze_samples(
        gaze_data, transformation_history, frame_width, frame_height
    )
    
    if not processed_data:
        print("No data was processed. Exiting.")
        return False
    
    # Step 3: Save final CSV
    save_final_csv(processed_data, output_csv_path)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Input gaze samples: {len(gaze_data)}")
    print(f"Output CSV records: {len(processed_data)}")
    print(f"Final CSV file: {output_csv_path}")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        exit(0)
    else:
        exit(1)
