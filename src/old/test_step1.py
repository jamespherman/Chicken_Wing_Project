"""
test_step1.py - Validation script for transformation history functionality

"""

import cv2
import numpy as np
import os


def test_transformation_history():
    """
    Test the transformation history file for correctness.
    """
    
    # File paths
    video_path = '/Users/sachitanand/Lab_PatrickMayo/Projects/Surgical_OpenCV/20231027T174922Z_ChickenWing/scenevideo.mp4'
    history_path = 'transformation_history.npy'
    
    print("="*60)
    print("TRANSFORMATION HISTORY VALIDATION TEST")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return False
        
    if not os.path.exists(history_path):
        print(f"ERROR: History file not found at {history_path}")
        return False
    
    # Load the transformation history
    try:
        transformation_history = np.load(history_path, allow_pickle=True)
        print(f"Successfully loaded transformation history from {history_path}")
    except Exception as e:
        print(f"ERROR: Failed to load transformation history: {e}")
        return False
    
    # Get video frame count
    try:
        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"Video has {total_video_frames} total frames")
    except Exception as e:
        print(f"ERROR: Failed to read video properties: {e}")
        return False
    
    print("\n" + "-"*60)
    print("TEST 1: COMPLETENESS TEST")
    print("-"*60)
    
    # Test 1: Completeness Test
    history_length = len(transformation_history)
    print(f"History records: {history_length}")
    print(f"Video frames: {total_video_frames}")
    
    if history_length == total_video_frames:
        print("PASS: History length matches video frame count")
        test1_pass = True
    else:
        print("FAIL: History length does not match video frame count")
        test1_pass = False
    
    print("\n" + "-"*60)
    print("TEST 2: INITIAL STATE TEST")
    print("-"*60)
    
    # Test 2: Initial State Test - check early frames for None homography
    test2_pass = True
    early_frames_to_check = min(20, history_length)  # Check first 20 frames or all if fewer
    
    none_count = 0
    for i in range(early_frames_to_check):
        frame_record = transformation_history[i]
        if frame_record['homography_matrix'] is None:
            none_count += 1
    
    print(f"Checked first {early_frames_to_check} frames")
    print(f"Frames with None homography: {none_count}")
    
    if none_count > 0:
        print("PASS: Found frames with None homography in early frames")
        test2_pass = True
    else:
        print("FAIL: No frames with None homography found in early frames")
        test2_pass = False
    
    print("\n" + "-"*60)
    print("TEST 3: GOOD FRAME TEST")
    print("-"*60)
    
    # Test 3: Good Frame Test - find first frame with valid homography
    first_good_frame_idx = None
    first_good_homography = None
    
    for i, frame_record in enumerate(transformation_history):
        if frame_record['homography_matrix'] is not None:
            first_good_frame_idx = i
            first_good_homography = frame_record['homography_matrix']
            break
    
    if first_good_frame_idx is not None:
        print(f"First good frame found at index: {first_good_frame_idx}")
        print(f"Homography shape: {first_good_homography.shape}")
        print(f"Homography dtype: {first_good_homography.dtype}")
        
        if first_good_homography.shape == (3, 3):
            print("PASS: First good frame has 3x3 homography matrix")
            test3_pass = True
        else:
            print("FAIL: First good frame does not have 3x3 homography matrix")
            test3_pass = False
    else:
        print("FAIL: No frames with valid homography found")
        test3_pass = False
        first_good_homography = None
    
    print("\n" + "-"*60)
    print("TEST 4: PERSISTENCE TEST")
    print("-"*60)
    
    # Test 4: Persistence Test - verify homography persists through subsequent frames
    test4_pass = False
    
    if first_good_homography is not None and first_good_frame_idx is not None:
        # Analyze homography patterns to understand persistence
        unique_matrices = []
        matrix_usage_counts = {}
        
        # Categorize all matrices
        for i, record in enumerate(transformation_history):
            if record['homography_matrix'] is not None:
                # Check if this matrix matches any we've seen before
                matrix_found = False
                for j, unique_matrix in enumerate(unique_matrices):
                    if np.allclose(record['homography_matrix'], unique_matrix, atol=1e-10):
                        matrix_usage_counts[j] = matrix_usage_counts.get(j, 0) + 1
                        matrix_found = True
                        break
                
                if not matrix_found:
                    # This is a new unique matrix
                    matrix_id = len(unique_matrices)
                    unique_matrices.append(record['homography_matrix'].copy())
                    matrix_usage_counts[matrix_id] = 1
        
        print(f"Total unique homography matrices found: {len(unique_matrices)}")
        matrices_used_multiple_times = sum(1 for count in matrix_usage_counts.values() if count > 1)
        print(f"Matrices used in multiple frames: {matrices_used_multiple_times}")
        
        # Expected behavior: Multiple unique matrices + some reused (persistence)
        if len(unique_matrices) > 10 and matrices_used_multiple_times > 0:
            print("PASS: Found evidence of both new calculations and persistence")
            test4_pass = True
        elif len(unique_matrices) <= 1:
            print("FAIL: Too few unique matrices (suggests only calculated once)")
            test4_pass = False
        else:
            print("UNCLEAR: Found multiple matrices but unclear persistence pattern")
            test4_pass = False
            
    else:
        print("FAIL: Cannot test persistence without a valid first good frame")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests_passed = sum([test1_pass, test2_pass, test3_pass, test4_pass])
    total_tests = 4
    
    print(f"Test 1 (Completeness): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Initial State): {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Good Frame): {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Persistence): {'PASS' if test4_pass else 'FAIL'}")
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("ALL TESTS PASSED! Transformation history is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the implementation.")
        return False


def display_sample_data():
    """
    Display some sample data from the transformation history for manual inspection.
    """
    
    history_path = 'transformation_history.npy'
    
    if not os.path.exists(history_path):
        print("History file not found for sample display.")
        return
    
    try:
        transformation_history = np.load(history_path, allow_pickle=True)
        
        print("\n" + "="*60)
        print("SAMPLE DATA INSPECTION")
        print("="*60)
        
        # Show first few frames
        print("First 5 frames:")
        for i in range(min(5, len(transformation_history))):
            record = transformation_history[i]
            matrix_status = "None" if record['homography_matrix'] is None else f"3x3 array"
            print(f"  Frame {record['frame_index']}: time={record['frame_time']}, H={matrix_status}")
        
        # Find and show first good frame
        for i, record in enumerate(transformation_history):
            if record['homography_matrix'] is not None:
                print(f"\nFirst good frame (Frame {record['frame_index']}):")
                print(f"  Time: {record['frame_time']}")
                print(f"  Homography matrix:")
                print(f"  {record['homography_matrix']}")
                break
                
    except Exception as e:
        print(f"Error displaying sample data: {e}")


if __name__ == "__main__":
    success = test_transformation_history()
    display_sample_data()
    
    if success:
        exit(0)
    else:
        exit(1)
