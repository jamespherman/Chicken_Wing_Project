# Codebase Structure

This document maps the folder structure and explains the purpose of key files.

## Root Directory

*   **`AGENTS.md`**: Instructions for AI agents working on this repo.
*   **`README.md`**: The main entry point and project overview.
*   **`config.json`**: Configuration settings for the analysis pipeline (generated on first run).
*   **`requirements.txt`**: Python dependencies.

## `src/` (Source Code)

The core logic of the application.

*   **`batch_process_with_heatmaps.py`**: **MAIN SCRIPT**. The orchestrator that runs the entire pipeline (correction -> CSV -> analysis -> visualization).
*   **`logging_config.py`**: Centralized logging setup.

### `src/processing/` (Data Pipeline)

Scripts that transform raw data into usable formats.

*   **`gaze_on_perspective_corrected_frames_refactored.py`**: Handles video processing, ArUco marker detection, and homography (Perspective Correction).
*   **`create_final_csv_refactored.py`**: Merges gaze, IMU, and video data into the final high-resolution CSV.
*   **`surgical_skill_analysis.py`**: (**Experimental/Legacy**) Contains the `IVTEventClassifier` and the `SurgicalSkillAnalyzer` class. Used for prototyping analysis ideas.
*   **`utils.py`**: Helper functions for geometry and data manipulation.

### `src/analysis/` (Scientific Analysis)

Scripts that interpret the data to produce metrics and insights.

*   **`adaptive_saccade_detector.py`**: The MAD-based algorithm for robust saccade detection.
*   **`whole_session_analysis.py`**: The main analysis engine calculating metrics for Goals A (Efficiency), B (Cognitive), C (Stability), and D (Strategy).
*   **`gaze_heatmap_analysis.py`**: Generates 2D heatmaps and scatter plots of gaze distribution.
*   **`pupil_luminance_kernel.py`**: Mathematical model for PLR (Pupillary Light Reflex) correction used in Goal B.
*   **`visualizations.py`**: Generates clinical charts (Radar plots, timelines).
*   **`cohort_analyzer.py`**: Aggregates data across multiple subjects (if enabled).

## `data/` (Data Storage)

*   **`raw/`**: Input directory for subject data.
*   **`processed/`**: Output directory for intermediate and final data files.

## `reports/` (Outputs)

*   **`figures/`**: Generated images (PNGs) of heatmaps, dashboards, and plots.
*   **`logs/`**: Text logs of the processing run and JSON files containing calculated metrics.
