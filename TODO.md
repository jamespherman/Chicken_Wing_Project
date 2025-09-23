# To-Do

This file lists potential optimizations and improvements for the Chicken Wing Surgical Cognition Analysis project.

## High Priority

- **[x] Optimize `find_active_transformation` function:** The current implementation in `src/processing/create_final_csv_refactored.py` uses a nested loop. This should be optimized to a single-pass (merge-like) operation since both gaze data and transformation history are sorted by time. (Already implemented)
- **[x] Parallelize video processing:** The frame-by-frame processing in `src/processing/gaze_on_perspective_corrected_frames_refactored.py` is a bottleneck. Implement parallel processing to speed up ArUco marker detection and perspective transformation. (Already implemented)

## Medium Priority

- **[x] Externalize configuration:** Move the hardcoded configuration from `src/batch_process_with_heatmaps.py` to a separate file (e.g., `config.json` or `config.yaml`) to allow for easier modification without changing the source code.
- **[x] Refactor `EnhancedBatchProcessor` class:** The `EnhancedBatchProcessor` class in `src/batch_process_with_heatmaps.py` is large. Break it down into smaller, more focused modules (e.g., for subject discovery, summary reporting) to improve modularity and readability.
- **[x] Use `logging` module:** Replace `print` statements with the built-in `logging` module for better control over log levels and output.
- **[x] Consolidate duplicated code:** The `transform_gaze_point` function is duplicated in two files. Move it to `src/processing/utils.py` to avoid redundancy.
- **[x] Pin dependencies:** The `requirements.txt` file should specify exact versions for all dependencies to ensure reproducible results.

## Low Priority

- **[x] Streamline data loading:** For very large datasets, loading all data into memory at once can be inefficient. Investigate streaming or chunking data in the processing scripts to reduce memory usage.
- **[x] Use pandas for CSV writing:** In `src/processing/create_final_csv_refactored.py`, consider using the pandas library to write the final CSV, which can simplify the code.
- **[x] Update `README.md`:** The `README.md` file mentions a non-existent `IMPROVEMENT_PLAN.md` file. This should be removed or updated.

## Future Improvements

- **[ ] Improve Test Coverage:** The current test suite (`src/test_batch_process.py`) is a high-level integration test that mocks the core processing functions. Add unit tests for the data processing logic in `src/processing/create_final_csv_refactored.py` and `src/processing/gaze_on_perspective_corrected_frames_refactored.py` to ensure correctness and prevent regressions.
- **[ ] Externalize Frame Dimensions:** The `frame_width` and `frame_height` are hardcoded in `src/processing/create_final_csv_refactored.py`. Move these to the `config.json` file to make the pipeline more flexible for different video resolutions.
- **[ ] Add Log Rotation/Cleanup:** The `reports/logs` directory accumulates log files from each run. Implement a log rotation or a cleanup strategy to manage the number of log files and save disk space.
- **[ ] Parameterize data filenames:** The filenames for the scene video (`scenevideo.mp4`) and gaze data (`gazedata.gz`) are hardcoded in `src/batch_process_with_heatmaps.py`. These should be moved to `config.json` to allow for more flexibility in data naming conventions.