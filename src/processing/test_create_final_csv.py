import pytest
import pandas as pd
import numpy as np
import json
import gzip
from unittest.mock import mock_open, patch
from ..processing.create_final_csv_refactored import (
    load_gaze_data_stream,
    load_transformation_history,
    process_gaze_stream,
    save_stream_to_csv,
    create_final_gaze_csv,
)

# Mock data for testing
@pytest.fixture
def mock_gaze_data():
    return [
        {"timestamp": 1.0, "data": {"gaze2d": [0.1, 0.2]}},
        {"timestamp": 2.0, "data": {"gaze2d": [0.3, 0.4]}},
        {"timestamp": 3.0, "data": {"gaze2d": [0.5, 0.6]}},
    ]

@pytest.fixture
def mock_transformation_history():
    return np.array([
        {'frame_index': 0, 'frame_time': 0.5, 'homography_matrix': np.eye(3)},
        {'frame_index': 1, 'frame_time': 1.5, 'homography_matrix': np.eye(3) * 2},
        {'frame_index': 2, 'frame_time': 2.5, 'homography_matrix': None}, # Invalid record
        {'frame_index': 3, 'frame_time': 3.5, 'homography_matrix': np.eye(3) * 3},
    ], dtype=object)


def test_load_gaze_data_stream(mock_gaze_data, tmp_path):
    """Test streaming gaze data from a gzipped file."""
    gaze_file = tmp_path / "gazedata.gz"
    with gzip.open(gaze_file, 'wt') as f:
        for sample in mock_gaze_data:
            f.write(json.dumps(sample) + '\n')

    stream = load_gaze_data_stream(gaze_file)
    loaded_data = list(stream)

    assert len(loaded_data) == len(mock_gaze_data)
    assert loaded_data[0]['timestamp'] == 1.0

def test_load_transformation_history(mock_transformation_history, tmp_path):
    """Test loading transformation history from a .npy file."""
    history_file = tmp_path / "history.npy"
    np.save(history_file, mock_transformation_history)

    history = load_transformation_history(history_file)

    assert history is not None
    assert len(history) == len(mock_transformation_history)
    assert history[0]['frame_index'] == 0

def test_process_gaze_stream(mock_gaze_data, mock_transformation_history):
    """Test the core logic of processing a gaze stream against transformation history."""
    gaze_stream = (s for s in mock_gaze_data)

    processed_stream = process_gaze_stream(gaze_stream, mock_transformation_history, 100, 100)
    processed_data = list(processed_stream)

    assert len(processed_data) == 3
    # Check first gaze sample - should be matched with the first valid transform
    assert processed_data[0]['active_frame_time'] == 0.5
    assert not np.isnan(processed_data[0]['transformed_gaze_x'])

    # Check second gaze sample - should be matched with the second valid transform
    assert processed_data[1]['active_frame_time'] == 1.5

    # Check third gaze sample - should be matched with the second valid transform (last valid one before timestamp 3.0)
    assert processed_data[2]['active_frame_time'] == 1.5


def test_save_stream_to_csv(tmp_path):
    """Test saving a stream of processed data to a CSV file."""
    processed_data = [
        {'gaze_timestamp': 1.0, 'transformed_gaze_x': 10, 'transformed_gaze_y': 20, 'active_frame_index': 1, 'active_frame_time': 0.5},
        {'gaze_timestamp': 2.0, 'transformed_gaze_x': np.nan, 'transformed_gaze_y': np.nan, 'active_frame_index': np.nan, 'active_frame_time': np.nan},
    ]
    output_csv_path = tmp_path / "output.csv"

    stream = (item for item in processed_data)
    stats = save_stream_to_csv(stream, output_csv_path)

    assert output_csv_path.exists()
    df = pd.read_csv(output_csv_path)
    assert len(df) == 2
    assert stats['total_records'] == 2
    assert stats['valid_transformations'] == 1
    assert stats['invalid_transformations'] == 1

@patch('src.processing.create_final_csv_refactored.load_gaze_data_stream')
@patch('src.processing.create_final_csv_refactored.load_transformation_history')
@patch('src.processing.create_final_csv_refactored.save_stream_to_csv')
def test_create_final_gaze_csv_integration(mock_save, mock_load_hist, mock_load_gaze, mock_gaze_data, mock_transformation_history):
    """Test the main orchestrator function."""
    mock_load_gaze.return_value = (s for s in mock_gaze_data)
    mock_load_hist.return_value = mock_transformation_history
    mock_save.return_value = {
        'total_records': 3,
        'valid_transformations': 2,
        'invalid_transformations': 1,
        'valid_percentage': 66.6
    }

    results = create_final_gaze_csv("dummy_gaze.gz", "dummy_hist.npy", "dummy_out.csv")

    assert results['success']
    assert results['output_csv_records'] == 3
    mock_load_gaze.assert_called_once()
    mock_load_hist.assert_called_once()
    mock_save.assert_called_once()
