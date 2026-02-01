# Chicken Wing Surgical Cognition Analysis

## Project Overview

This project quantifies surgical skill by analyzing eye movement dynamics (oculometrics) and head kinematics. It processes raw data from Tobii Pro Glasses 3 (gaze, video, IMU), stabilizes the view using ArUco markers, and computes advanced metrics for **Oculometric Efficiency**, **Cognitive Load**, **Motor Stability**, and **Visual Strategy**.

**Goal:** To distinguish between expert and novice surgeons based on their cognitive and visual behaviors during a dissection task.

## 📚 Documentation

We have detailed documentation available in the `docs/` directory:

*   **[Usage Guide](docs/usage.md)**: How to install, configure, and run the code. **Start here.**
*   **[Algorithms & Methodology](docs/algorithms.md)**: Deep dive into Perspective Correction, Saccade Detection (MAD vs. I-VT), and Physics-based metrics.
*   **[Analysis Goals](docs/analysis.md)**: Explanation of the scientific metrics (Goals A-D) and visualizations.
*   **[Codebase Structure](docs/codebase_structure.md)**: A map of the files and folders in this repository.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis

Place your subject data in `data/raw/` and run:

```bash
python3 src/batch_process_with_heatmaps.py
```

### 3. View Results

Check `reports/figures/` for heatmaps and dashboards, and `data/processed/` for the high-resolution CSV data.

## Features

*   **Perspective Correction**: Maps gaze to a static reference frame using ArUco markers.
*   **Adaptive Saccade Detection**: Robustly identifies eye movements using MAD-based thresholding.
*   **Cognitive Load Indexing**: Separates mental effort from light-reflex pupil dilation.
*   **Visual Strategy Classification**: Detects whether the surgeon is looking at tools or tissue.
*   **Automated Visualization**: Generates heatmaps, scanpath plots, and clinical skill dashboards.
