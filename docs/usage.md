# Usage Guide

This guide provides comprehensive instructions on how to use the Chicken Wing Surgical Cognition Analysis software.

## Table of Contents

1. [Installation](#installation)
2. [Data Organization](#data-organization)
3. [Configuration](#configuration)
4. [Running the Analysis](#running-the-analysis)
5. [Understanding Outputs](#understanding-outputs)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- **Python 3.8+**: Ensure you have Python installed.
- **Git**: For cloning the repository.
- **Sandbox Environment**: If running in an isolated environment (like this one), dependencies may need to be installed in a virtual environment.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This installs necessary packages like `opencv-python`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, and `tqdm`.

## Data Organization

The system expects a specific directory structure to discover and process subject data automatically.

```
project_root/
├── data/
│   ├── raw/                  <-- Put your raw subject folders here
│   │   ├── Subject001/
│   │   │   ├── gazedata.gz   <-- Raw Tobii gaze data
│   │   │   ├── imudata.gz    <-- Raw Tobii IMU data (optional)
│   │   │   ├── scenevideo.mp4 <-- Scene camera video
│   │   │   └── ...
│   │   ├── Subject002/
│   │   └── ...
```

*   **`gazedata.gz`**: Compressed JSON lines file containing eye tracking data.
*   **`scenevideo.mp4`**: The video recording from the eye tracker's scene camera.
*   **`imudata.gz`**: (Optional) Compressed JSON lines file containing accelerometer/gyroscope data.

## Configuration

The analysis is controlled by a `config.json` file in the root directory. If it doesn't exist, the script will generate a default one.

### Key Configuration Options

```json
{
    "input_base_dir": "data/raw",         // Directory containing subject folders
    "subjects_to_skip": ["TestSubject"],  // List of folder names to ignore
    "skip_existing": true,                // Skip subjects that already have output files
    "generate_heatmaps": true,            // Generate visual heatmaps
    "run_whole_session_analysis": true,   // Run the physics-based skill analysis

    "processing_options": {
        "use_preselected_parameters": false, // Use tuned ArUco parameters
        "use_frame_preprocessing": false,    // Enhance contrast before marker detection
        "show_video": false                  // Show video window during processing (local only)
    },

    "target_markers": [13, 14, 15, 16],   // ArUco marker IDs defining the workspace
    "output_width": 1000,                 // Width of the corrected output video
    "output_height": 606                  // Height of the corrected output video
}
```

## Running the Analysis

To process all subjects in the `data/raw` directory:

```bash
python3 src/batch_process_with_heatmaps.py
```

### What happens when you run this?

1.  **Discovery**: The script scans `data/raw` for valid subject folders.
2.  **Processing (Per Subject)**:
    *   **Perspective Correction**: It reads the video, detects ArUco markers, and calculates the perspective transformation (homography) to stabilize the image.
    *   **Data Synchronization**: It maps gaze points to the stabilized frame and extracts physics metrics (velocity, head movement).
    *   **Analysis**: It computes surgical skill metrics (Oculometric efficiency, cognitive load, etc.).
    *   **Visualization**: It generates heatmaps and clinical dashboards.
3.  **Reporting**: A summary report is generated in `reports/logs`.

## Understanding Outputs

After processing, results are organized in the project root:

```
project_root/
├── data/
│   └── processed/
│       └── Subject001/
│           ├── Subject001_final_gaze_data.csv       <-- The "Golden Source" of data
│           ├── Subject001_gaze_corrected_video.mp4  <-- Stabilized video with gaze overlay
│           ├── Subject001_transformation_history.npy <-- Matrix history for debugging
│           └── ...
├── reports/
│   ├── figures/
│   │   ├── Subject001_dashboard.png                 <-- Heatmap + marginal plots
│   │   ├── Subject001_viz_cognitive_fingerprint.png <-- Radar chart of skill
│   │   └── ...
│   └── logs/
│       ├── Subject001_processing_log.txt            <-- Detailed log
│       └── Subject001_whole_session_analysis.json   <-- Calculated metrics (Goal A-D)
```

### `final_gaze_data.csv` Columns
*   `gaze_timestamp`: Time in seconds.
*   `transformed_gaze_x/y`: Gaze coordinates in the stabilized reference frame (pixels).
*   `angular_velocity_deg_s`: Eye movement speed (degrees/second).
*   `pupil_diameter_avg`: Average pupil size (mm).
*   `head_gyro_x/y/z`: Head rotation velocity.
*   `frame_luminance`: Brightness of the scene (0-255).

## Troubleshooting

*   **"No valid subject folders found"**: Check that your `data/raw` structure matches the [Data Organization](#data-organization) section.
*   **Markers not detected**:
    *   Ensure the printed ArUco markers (IDs 13-16) are visible in the video.
    *   Try enabling `"use_frame_preprocessing": true` in `config.json`.
*   **Memory errors**: If processing large videos, ensure you have sufficient RAM. The script uses multiprocessing; you can reduce the number of workers in the code if necessary.
