import pytest
import numpy as np
import cv2
import json
import gzip
from unittest.mock import patch, MagicMock

from ..processing.gaze_on_perspective_corrected_frames_refactored import (
    find_outer_points,
    find_and_order_average_points,
    _process_frame_chunk,
    load_gaze_data,
    extract_timestamps_and_gaze_positions,
)

@pytest.fixture
def aruco_marker_data():
    """Provides mock corners and IDs for ArUco markers."""
    corners = [
        np.array([[[10, 10], [30, 10], [30, 30], [10, 30]]], dtype=np.float32), # ID 13
        np.array([[[70, 10], [90, 10], [90, 30], [70, 30]]], dtype=np.float32), # ID 14
        np.array([[[70, 70], [90, 70], [90, 90], [70, 90]]], dtype=np.float32), # ID 15
        np.array([[[10, 70], [30, 70], [30, 90], [10, 90]]], dtype=np.float32), # ID 16
    ]
    ids = np.array([[13], [14], [15], [16]])
    valid_ids = [13, 14, 15, 16]
    return corners, ids, valid_ids

def test_find_and_order_average_points(aruco_marker_data):
    """Tests that the average points of markers are correctly found and ordered."""
    corners, ids, valid_ids = aruco_marker_data

    ordered_points = find_and_order_average_points(corners, ids, valid_ids)

    assert ordered_points is not None
    assert ordered_points.shape == (4, 2)
    # Expected averages: (20,20), (80,20), (80,80), (20,80)
    # After ordering: top-left, top-right, bottom-right, bottom-left
    expected = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)
    np.testing.assert_allclose(ordered_points, expected)

def test_find_outer_points(aruco_marker_data):
    """Tests that the outer points (convex hull) are correctly found and ordered."""
    corners, ids, valid_ids = aruco_marker_data

    ordered_points = find_outer_points(corners, ids, valid_ids)

    assert ordered_points is not None
    assert ordered_points.shape == (4, 2)
    # Expected outer corners: (10,10), (90,10), (90,90), (10,90)
    # After ordering: top-left, top-right, bottom-right, bottom-left
    expected = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    np.testing.assert_allclose(ordered_points, expected)


@pytest.fixture
def mock_video_and_gaze(tmp_path):
    """Creates a dummy video file and a mock gaze data file."""
    video_path = tmp_path / "test_video.mp4"
    gaze_path = tmp_path / "gazedata.gz"

    # Create a dummy video file (e.g., 10 frames of black)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10, (100, 100))
    for _ in range(10):
        out.write(np.zeros((100, 100, 3), dtype=np.uint8))
    out.release()

    # Create mock gaze data
    gaze_data = [
        {"timestamp": 0.15, "data": {"gaze2d": [0.5, 0.5]}},
        {"timestamp": 0.35, "data": {"gaze2d": [0.6, 0.6]}},
    ]
    with gzip.open(gaze_path, 'wt') as f:
        for sample in gaze_data:
            f.write(json.dumps(sample) + '\n')

    return str(video_path), str(gaze_path)


def test_process_frame_chunk(mock_video_and_gaze, aruco_marker_data):
    """Test the _process_frame_chunk worker function."""
    video_path, gaze_path = mock_video_and_gaze
    mock_corners, mock_ids, valid_ids = aruco_marker_data

    # Mock the aruco detector to return our predefined markers
    with patch('cv2.aruco.ArucoDetector.detectMarkers') as mock_detect:
        mock_detect.return_value = (mock_corners, mock_ids, None)

        gaze_data = load_gaze_data(gaze_path)

        args = (
            video_path,      # video_path
            0,               # chunk_index
            0,               # start_frame
            5,               # end_frame
            gaze_data,       # gaze_data
            100, 100,        # frame_width, frame_height
            200, 200,        # output_width, output_height
            valid_ids,       # target_markers
            True, True, True # use_preselected_parameters, use_frame_preprocessing, use_outer_points
        )

        result = _process_frame_chunk(args)

        assert result is not None
        chunk_index, video_frames, csv_data, trans_hist, stats = result

        assert chunk_index == 0
        assert len(video_frames) == 5
        assert len(csv_data) == 5
        assert len(trans_hist) == 5

        # Check stats
        assert stats['total_frames'] == 5
        assert stats['frames_with_markers'] == 5
        assert stats['frames_with_valid_homography'] == 5
        # Gaze is interpolated, so we expect gaze points
        assert stats['frames_with_gaze'] > 0

        # Check that homography was stored
        assert trans_hist[0]['homography_matrix'] is not None
        # Check that gaze was transformed
        assert np.isnan(csv_data[1][2]) # frame at index 1 (time 0.1) is before first gaze, should be nan
        assert not np.isnan(csv_data[2][2]) # frame at index 2 (time 0.2) should have gaze
        assert not np.isnan(csv_data[3][2]) # frame at index 3 (time 0.3) should have gaze
