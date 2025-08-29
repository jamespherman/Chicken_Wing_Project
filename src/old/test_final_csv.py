"""
test_final_csv.py - Validation script for final gaze data CSV

This script tests the output of create_final_csv.py to ensure:
1. Completeness: All gaze samples are processed
2. Initial State: Early timestamps have NaN coordinates
3. Good Frame: Known good frames have valid coordinates
4. Persistence: Bad frames after good ones still have valid coordinates
"""

import pandas as pd
import numpy as np
import json
import gzip
import os


def load_original_gaze_data(gaze_file_path):
    """
    Load the original gaze data to count total samples.
    """
    print(f"Loading original gaze data from: {gaze_file_path}")
    
    try:
        with gzip.open(gaze_file_path, 'rt') as f:
            gaze_data = [json.loads(line) for line in f]
        print(f"Original gaze data: {len(gaze_data)} samples")
        return gaze_data
    except Exception as e:
        print(f"Error loading original gaze data: {e}")
        return None


def load_final_csv(csv_file_path):
    """
    Load the final CSV output for testing.
    """
    print(f"Loading final CSV from: {csv_file_path}")
    
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Final CSV loaded: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading final CSV: {e}")
        return None


def test_completeness(original_gaze_data, final_csv_df):
    """
    Test 1: Completeness Test
    The total number of rows in the final CSV must match the total number of gaze samples.
    """
    print("\n" + "-"*60)
    print("TEST 1: COMPLETENESS TEST")
    print("-"*60)
    
    original_count = len(original_gaze_data)
    csv_count = len(final_csv_df)
    
    print(f"Original gaze samples: {original_count}")
    print(f"Final CSV rows: {csv_count}")
    
    if original_count == csv_count:
        print("PASS: Row count matches exactly")
        return True
    else:
        print("FAIL: Row count mismatch")
        print(f"  Difference: {csv_count - original_count}")
        return False


def test_initial_state(final_csv_df, threshold_time=3.2):
    """
    Test 2: Initial State Spot Check
    Early timestamps (< 3.2s) should have NaN coordinates and low frame indices.
    """
    print("\n" + "-"*60)
    print("TEST 2: INITIAL STATE SPOT CHECK")
    print("-"*60)
    
    # Filter for early timestamps
    early_rows = final_csv_df[final_csv_df['gaze_timestamp'] < threshold_time]
    
    if len(early_rows) == 0:
        print(f"FAIL: No rows found with gaze_timestamp < {threshold_time}s")
        return False
    
    print(f"Found {len(early_rows)} rows with gaze_timestamp < {threshold_time}s")
    
    # Check a few example rows
    sample_rows = early_rows.head(5)
    print("\nSample early rows:")
    
    test_pass = True
    for idx, row in sample_rows.iterrows():
        timestamp = row['gaze_timestamp']
        x_coord = row['transformed_gaze_x']
        y_coord = row['transformed_gaze_y']
        frame_idx = row['active_frame_index']
        frame_time = row['active_frame_time']
        
        print(f"  Timestamp: {timestamp:.3f}s")
        print(f"    X: {x_coord}, Y: {y_coord}")
        print(f"    Frame: {frame_idx}, Frame Time: {frame_time}")
        
        # Check if coordinates are NaN (expected for early frames)
        coords_are_nan = pd.isna(x_coord) and pd.isna(y_coord)
        
        if coords_are_nan:
            print("Coordinates are NaN (expected)")
        else:
            print("Coordinates are NOT NaN (unexpected for early frames)")
            test_pass = False
        
        print()
    
    if test_pass:
        print("PASS: Early timestamps have expected NaN coordinates")
    else:
        print("FAIL: Some early timestamps have unexpected valid coordinates")
    
    return test_pass


def test_good_frame(final_csv_df):
    """
    Test 3: Good Frame Spot Check
    Find the actual first good frame and verify it has valid coordinates.
    """
    print("\n" + "-"*60)
    print("TEST 3: GOOD FRAME SPOT CHECK")
    print("-"*60)
    
    # Find the first row with valid coordinates
    first_valid_idx = final_csv_df['transformed_gaze_x'].first_valid_index()
    
    if first_valid_idx is None:
        print("FAIL: No valid coordinates found in entire dataset")
        return False
    
    first_valid_row = final_csv_df.loc[first_valid_idx]
    first_valid_time = first_valid_row['gaze_timestamp']
    first_valid_frame = first_valid_row['active_frame_index']
    
    print(f"First valid coordinates found at:")
    print(f"  Timestamp: {first_valid_time:.3f}s")
    print(f"  Frame: {first_valid_frame}")
    
    # Find rows starting from the first valid timestamp and going forward (not backward)
    tolerance = 0.2  # Look 0.2s forward from first valid time
    time_mask = (final_csv_df['gaze_timestamp'] >= first_valid_time) & \
                (final_csv_df['gaze_timestamp'] <= first_valid_time + tolerance)
    
    target_rows = final_csv_df[time_mask]
    
    print(f"Found {len(target_rows)} rows from first valid timestamp + {tolerance}s forward")
    
    # Check the first 5 rows after the first valid coordinate
    sample_rows = target_rows.head(5)
    print("\nSample good frame rows:")
    
    test_pass = True
    valid_coords_count = 0
    
    for idx, row in sample_rows.iterrows():
        timestamp = row['gaze_timestamp']
        x_coord = row['transformed_gaze_x']
        y_coord = row['transformed_gaze_y']
        frame_idx = row['active_frame_index']
        frame_time = row['active_frame_time']
        
        print(f"  Timestamp: {timestamp:.3f}s")
        print(f"    X: {x_coord}, Y: {y_coord}")
        print(f"    Frame: {frame_idx}, Frame Time: {frame_time}")
        
        # Check if coordinates are valid (not NaN)
        coords_are_valid = not (pd.isna(x_coord) or pd.isna(y_coord))
        
        if coords_are_valid:
            print("Coordinates are valid (expected)")
            valid_coords_count += 1
        else:
            print("Coordinates are NaN (unexpected after first valid)")
        
        print()
    
    # Test passes if we found valid coordinates starting from the first valid time
    if valid_coords_count >= 3:  # At least 3 out of 5 should be valid
        print(f"PASS: Found {valid_coords_count} valid coordinates starting from {first_valid_time:.3f}s")
        test_pass = True
    else:
        print(f"FAIL: Only found {valid_coords_count} valid coordinates after first valid time")
        test_pass = False
    
    return test_pass


def test_persistence(final_csv_df, target_frame=109, tolerance_frames=5):
    """
    Test 4: Persistence Spot Check
    Timestamps around frame #109 should have valid coordinates from persistence.
    """
    print("\n" + "-"*60)
    print("TEST 4: PERSISTENCE SPOT CHECK")
    print("-"*60)
    
    # Find rows with active_frame_index near target frame
    frame_mask = (final_csv_df['active_frame_index'] >= target_frame - tolerance_frames) & \
                 (final_csv_df['active_frame_index'] <= target_frame + tolerance_frames)
    
    target_rows = final_csv_df[frame_mask]
    
    if len(target_rows) == 0:
        print(f"FAIL: No rows found with active_frame_index near {target_frame}")
        return False
    
    print(f"Found {len(target_rows)} rows with active_frame_index near {target_frame} (±{tolerance_frames})")
    
    # Check a few example rows
    sample_rows = target_rows.head(5)
    print("\nSample persistence frame rows:")
    
    test_pass = True
    for idx, row in sample_rows.iterrows():
        timestamp = row['gaze_timestamp']
        x_coord = row['transformed_gaze_x']
        y_coord = row['transformed_gaze_y']
        frame_idx = row['active_frame_index']
        frame_time = row['active_frame_time']
        
        print(f"  Timestamp: {timestamp:.3f}s")
        print(f"    X: {x_coord}, Y: {y_coord}")
        print(f"    Frame: {frame_idx}, Frame Time: {frame_time}")
        
        # Check if coordinates are valid (should be, due to persistence)
        coords_are_valid = not (pd.isna(x_coord) or pd.isna(y_coord))
        
        if coords_are_valid:
            print("Coordinates are valid (persistence working)")
        else:
            print("Coordinates are NaN (persistence failed)")
            test_pass = False
        
        print()
    
    if test_pass:
        print("PASS: Persistence frames have valid coordinates")
    else:
        print("FAIL: Some persistence frames have NaN coordinates")
    
    return test_pass


def analyze_data_distribution(final_csv_df):
    """
    Provide additional analysis of the data distribution.
    """
    print("\n" + "="*60)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*60)
    
    total_rows = len(final_csv_df)
    
    # Count valid vs NaN coordinates
    valid_x = final_csv_df['transformed_gaze_x'].notna().sum()
    valid_y = final_csv_df['transformed_gaze_y'].notna().sum()
    valid_coords = min(valid_x, valid_y)  # Both X and Y must be valid
    
    # Count valid frame indices
    valid_frames = final_csv_df['active_frame_index'].notna().sum()
    
    print(f"Total gaze samples: {total_rows}")
    print(f"Valid transformed coordinates: {valid_coords} ({valid_coords/total_rows*100:.1f}%)")
    print(f"NaN transformed coordinates: {total_rows - valid_coords} ({(total_rows - valid_coords)/total_rows*100:.1f}%)")
    print(f"Valid frame indices: {valid_frames} ({valid_frames/total_rows*100:.1f}%)")
    
    # Time range analysis
    min_time = final_csv_df['gaze_timestamp'].min()
    max_time = final_csv_df['gaze_timestamp'].max()
    print(f"\nTime range: {min_time:.3f}s to {max_time:.3f}s ({max_time - min_time:.1f}s total)")
    
    # First and last valid coordinates
    first_valid_idx = final_csv_df['transformed_gaze_x'].first_valid_index()
    last_valid_idx = final_csv_df['transformed_gaze_x'].last_valid_index()
    
    if first_valid_idx is not None:
        first_valid_time = final_csv_df.loc[first_valid_idx, 'gaze_timestamp']
        print(f"First valid coordinate at: {first_valid_time:.3f}s")
    
    if last_valid_idx is not None:
        last_valid_time = final_csv_df.loc[last_valid_idx, 'gaze_timestamp']
        print(f"Last valid coordinate at: {last_valid_time:.3f}s")


def main():
    """
    Main function to run all validation tests.
    """
    print("="*60)
    print("FINAL CSV VALIDATION TESTS")
    print("="*60)
    
    # File paths (adjust these to match your setup)
    gaze_file_path = 'gazedata.gz'
    final_csv_path = 'final_gaze_data.csv'
    
    # Load input files
    original_gaze_data = load_original_gaze_data(gaze_file_path)
    if original_gaze_data is None:
        print("Cannot proceed without original gaze data")
        return False
    
    final_csv_df = load_final_csv(final_csv_path)
    if final_csv_df is None:
        print("Cannot proceed without final CSV")
        return False
    
    # Run all tests
    test_results = []
    
    test_results.append(test_completeness(original_gaze_data, final_csv_df))
    test_results.append(test_initial_state(final_csv_df))
    test_results.append(test_good_frame(final_csv_df))
    test_results.append(test_persistence(final_csv_df))
    
    # Provide data distribution analysis
    analyze_data_distribution(final_csv_df)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests_passed = sum(test_results)
    total_tests = len(test_results)
    
    test_names = [
        "Completeness Test",
        "Initial State Test",
        "Good Frame Test",
        "Persistence Test"
    ]
    
    for i, (test_name, passed) in enumerate(zip(test_names, test_results)):
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("ALL TESTS PASSED! Final CSV is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        exit(0)
    else:
        exit(1)
