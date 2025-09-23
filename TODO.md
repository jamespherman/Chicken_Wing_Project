# To-Do

This file lists potential optimizations and improvements for the Chicken Wing Surgical Cognition Analysis project.

## High Priority

- **[x] Improve Test Coverage:** The current test suite (`src/test_batch_process.py`) is a high-level integration test that mocks the core processing functions. Add unit tests for the data processing logic in `src/processing/create_final_csv_refactored.py` and `src/processing/gaze_on_perspective_corrected_frames_refactored.py` to ensure correctness and prevent regressions.

## Medium Priority

- **[ ] Externalize Frame Dimensions:** The `frame_width` and `frame_height` are hardcoded in `src/processing/create_final_csv_refactored.py`. Move these to the `config.json` file to make the pipeline more flexible for different video resolutions.
- **[ ] Parameterize data filenames:** The filenames for the scene video (`scenevideo.mp4`) and gaze data (`gazedata.gz`) are hardcoded in `src/batch_process_with_heatmaps.py`. These should be moved to `config.json` to allow for more flexibility in data naming conventions.

## Low Priority

- **[ ] Add Log Rotation/Cleanup:** The `reports/logs` directory accumulates log files from each run. Implement a log rotation or a cleanup strategy to manage the number of log files and save disk space.
- **[ ] Update `README.md` configuration section:** The `README.md` file incorrectly states that configuration is done in the `main()` function of `src/batch_process_with_heatmaps.py`. It should be updated to reflect that configuration is handled through `config.json`.