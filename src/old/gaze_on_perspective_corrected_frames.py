import cv2
import numpy as np
import os
import gzip
import json
import csv
from tqdm import tqdm  # For progress bar


############################################
# Preprocessing toggle features
############################################

# Function to enhance contrast and sharpness
def enhance_contrast_and_sharpness(frame):
    """
    Enhance contrast and sharpness of the input frame.
    """
    # Convert to grayscale for contrast enhancement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)

    # Convert back to BGR for sharpening
    enhanced_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    # Apply a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_frame, -1, kernel)

    return sharpened

# Apply preselected detector parameters
def apply_preselected_parameters(parameters):
    """
    Apply a predefined set of ArUco detector parameters.
    """
    # Adaptive Thresholding
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 80
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.adaptiveThreshConstant = 0

    # Marker Size Filtering
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 2

    # Contour Approximation
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.minCornerDistanceRate = 0.18

    # Corner Refinement - handle version differences
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 25
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01

    return parameters

# Find the most outer points from all marker corners (from Code B)
def find_outer_points(corners, ids, valid_ids):
    """
    Find the most outer points from all marker corners.
    """
    all_points = []
    for corner, marker_id in zip(corners, ids.flatten()):
        if marker_id in valid_ids:
            all_points.extend(corner[0])  # Append the 4 corners of each marker

    if len(all_points) == 0:
        return None  # No valid markers found

    all_points = np.array(all_points, dtype="float32")

    # Compute the convex hull to find the outer boundary
    if all_points.shape[0] < 4:  # Need at least 4 points for a hull
        return None

    hull = cv2.convexHull(all_points)

    if len(hull) >= 4:
        # Ensure consistent ordering
        ordered_points = order_points(hull.squeeze())
        return ordered_points
    return None

############################################
# Gaze dot plot functions
############################################

# Load gaze data
def load_gaze_data(gaze_file_path):
    with gzip.open(gaze_file_path, 'rt') as f:
        gaze_data = [json.loads(line) for line in f]
    return gaze_data

# Extract timestamps and gaze positions with alignment
def extract_timestamps_and_gaze_positions(gaze_data):
    timestamps = []
    gaze_positions = []
    for sample in gaze_data:
        timestamps.append(sample['timestamp'])
        if 'gaze2d' in sample['data']:
            gaze_positions.append(sample['data']['gaze2d'])
        else:
            gaze_positions.append([np.nan, np.nan])  # Placeholder for invalid gaze2d
    return timestamps, gaze_positions

# Function to apply homography to gaze point
def transform_gaze_point(gaze_point, homography_matrix, frame_width, frame_height):
    if any(np.isnan(gaze_point)):
        return None  # Invalid gaze point

    # Convert normalized gaze2d to pixel coordinates
    gaze_x = int(gaze_point[0] * frame_width)
    gaze_y = int(gaze_point[1] * frame_height)

    # Apply homography transformation
    original_point = np.array([[gaze_x, gaze_y]], dtype="float32")
    transformed_point = cv2.perspectiveTransform(np.array([original_point]), homography_matrix)

    # Return transformed point
    transformed_x, transformed_y = transformed_point[0][0]
    return int(transformed_x), int(transformed_y)

# Order points as top-left, top-right, bottom-right, bottom-left
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

# Find and order average points (already present in Code A)
def find_and_order_average_points(corners, ids, valid_ids):
    marker_points = []
    for corner, marker_id in zip(corners, ids.flatten()):
        if marker_id in valid_ids:
            avg_point = np.mean(corner[0], axis=0)  # Average the 4 corners
            marker_points.append(avg_point)
    if len(marker_points) != 4:
        return None  # Ensure exactly 4 valid markers
    return order_points(np.array(marker_points, dtype="float32"))

# Linearly interpolate gaze data
def interpolate_gaze(timestamps, gaze_positions, frame_time):
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
# Toggles Features
############################################
USE_PRESELECTED_PARAMETERS = False
USE_FRAME_PREPROCESSING = False
USE_OUTER_POINTS = False  # Toggle between using outer points or centers for perspective correction


############################################
# Main Processing Code
############################################

# Initialize video capture
folder_path = '/Users/sachitanand/Lab_PatrickMayo/Projects/Surgical_OpenCV/20231027T174922Z_ChickenWing'
file_name = 'scenevideo.mp4'
gaze_file_path = os.path.join(folder_path, "gazedata.gz")
full_file_path = os.path.join(folder_path, file_name)
cap = cv2.VideoCapture(full_file_path)

# Output settings
output_path = 'gaze_corrected_video_with_features.mp4'
csv_output_path = 'gaze_output.csv'  # CSV output file
transformation_history_path = 'transformation_history.npy'  # New output file for transformation history
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_width = 1000
output_height = int(output_width / 1.65)
out = cv2.VideoWriter(output_path, fourcc, 25.0, (output_width, output_height))

# Load gaze data
gaze_data = load_gaze_data(gaze_file_path)
timestamps, gaze_positions = extract_timestamps_and_gaze_positions(gaze_data)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
    print(f"Error: Unable to open video file at {full_file_path}")
    exit()
else:
    print(f"Processing video from: {full_file_path}")

# Initialize CSV file
csv_file = open(csv_output_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame_index', 'frame_time', 'transformed_gaze_x', 'transformed_gaze_y'])  # Header

# Initialize transformation history tracking
transformation_history = []  # List to store frame-by-frame analysis results
persistent_homography = None  # Variable to hold the most recent valid homography matrix

print("Processing video with gaze points and saving CSV data...")
frame_index = 0
with tqdm(total=total_frames, unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = frame_index / fps
        
        # Initialize gaze coordinates as NaN (will be used if no valid gaze point is found)
        transformed_gaze_x = np.nan
        transformed_gaze_y = np.nan

        # Preprocess frame if enabled
        if USE_FRAME_PREPROCESSING:
            preprocessed_frame = enhance_contrast_and_sharpness(frame)
        else:
            preprocessed_frame = frame

        # Detect ArUco markers
        gray = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        # Apply preselected parameters if enabled
        if USE_PRESELECTED_PARAMETERS:
            parameters = apply_preselected_parameters(parameters)

        # Try newer OpenCV version first, fallback to older version
        try:
            # For OpenCV 4.7+
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # For older OpenCV versions
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Check if we can calculate a new homography matrix
        new_homography_calculated = False
        if ids is not None and len(ids) > 0:
            target_markers = set([13, 14, 15, 16])
            detected_markers = set(ids.flatten())
            if target_markers.issubset(detected_markers):
                # Select method to find source points
                if USE_OUTER_POINTS:
                    src_points = find_outer_points(corners, ids, valid_ids=list(target_markers))
                else:
                    src_points = find_and_order_average_points(corners, ids, valid_ids=list(target_markers))

                if src_points is not None:
                    # Define desired corners for perspective correction
                    desired_corners = np.array([
                        [0, 0],
                        [output_width - 1, 0],
                        [output_width - 1, output_height - 1],
                        [0, output_height - 1]
                    ], dtype="float32")

                    # Compute homography matrix and update persistent matrix
                    # Calculate new homography every time markers are detected for accuracy
                    H, _ = cv2.findHomography(src_points, desired_corners)
                    persistent_homography = H.copy()  # Update the persistent homography
                    new_homography_calculated = True

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
                        # Store the coordinates for CSV output
                        transformed_gaze_x = tx
                        transformed_gaze_y = ty
                        # Draw the gaze point on the video
                        cv2.circle(corrected_frame, (tx, ty), 15, (0, 0, 255), -1)

            out.write(corrected_frame)
            cv2.imshow("Corrected Perspective with Gaze", corrected_frame)

        # Write CSV row for this frame (regardless of whether gaze point was found)
        csv_writer.writerow([frame_index, round(frame_time, 3), transformed_gaze_x, transformed_gaze_y])

        frame_index += 1
        pbar.update(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close files
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

# Save the complete transformation history to file
np.save(transformation_history_path, transformation_history, allow_pickle=True)

print("Processing complete!")
print(f"Video saved to: {output_path}")
print(f"Gaze data saved to: {csv_output_path}")
print(f"Transformation history saved to: {transformation_history_path}")
print(f"Total frames processed: {len(transformation_history)}")
print(f"Frames with valid homography: {sum(1 for record in transformation_history if record['homography_matrix'] is not None)}")
