import pytest
from unittest.mock import patch, mock_open
import json
from .batch_process_with_heatmaps import main as batch_main

@pytest.fixture
def mock_config_file(tmp_path):
    config = {
        "input_base_dir": "data/raw",
        "subjects_to_skip": [],
        "subject_folder_pattern": "*",
        "skip_existing": True,
        "generate_heatmaps": True,
        "create_summary_report": True,
        "video_filename": "scenevideo.mp4",
        "gaze_filename": "gazedata.gz",
        "heatmap_config": {
            "figure_size": [12, 8],
            "dpi": 300,
            "color_scheme": "viridis"
        },
        "processing_options": {
            "show_video": False
        }
    }
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return config_path

@patch('pathlib.Path.exists')
@patch('src.batch_process_with_heatmaps.discover_subject_folders')
@patch('src.batch_process_with_heatmaps.process_gaze_with_perspective_correction')
@patch('src.batch_process_with_heatmaps.create_final_gaze_csv')
@patch('src.batch_process_with_heatmaps.GazeHeatmapAnalyzer')
def test_main_runs_without_errors(mock_analyzer, mock_create_csv, mock_process_gaze, mock_discover, mock_exists, mock_config_file):
    # This test checks if the main function runs without raising exceptions.
    # It does not check for correctness of the output.

    # Mock the file system
    mock_exists.return_value = True
    mock_discover.return_value = ([mock_config_file.parent], 0)

    # Mock the processing functions to return successful results
    mock_process_gaze.return_value = {'frames_with_valid_homography': 1}
    mock_create_csv.return_value = {'success': True, 'valid_transformations': 1, 'valid_percentage': 100.0}
    mock_analyzer.return_value.analyze_subject.return_value = {'success': True, 'visualizations_created': [], 'statistics': {}}

    with patch('builtins.open', mock_open(read_data=mock_config_file.read_text())) as mock_file:
        result = batch_main()
        assert result is True, "main function should return True on success"
