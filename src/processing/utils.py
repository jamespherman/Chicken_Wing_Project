import numpy as np

def order_points(pts):
    """
    Order points as top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect


import cv2
import logging

# Configure a logger for this module
logger = logging.getLogger(__name__)

def transform_gaze_point(gaze_point, homography_matrix, frame_width=1920, frame_height=1080):
    """
    Apply homography transformation to a gaze point.

    Args:
        gaze_point (list): Normalized gaze coordinates [x, y] (0-1 range)
        homography_matrix (numpy.ndarray): 3x3 homography matrix
        frame_width (int): Original frame width in pixels
        frame_height (int): Original frame height in pixels

    Returns:
        tuple: (transformed_x, transformed_y) or (NaN, NaN) if invalid
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
        logger.warning(f"Warning: Error transforming gaze point {gaze_point}: {e}")
        return (np.nan, np.nan)
