Here is the comprehensive implementation plan for the visualization phase. This document is designed to be handed directly to your coding agent. It specifies exactly how to transform the physics-based data you just generated into the specific clinical insights we discussed.

You can save this content as `docs/2026-01-24_visualization_implementation_plan.md`.

***

# 2026-01-24 Visualization Implementation Plan

## Overview
Now that the "Physics Update" is complete and the data pipeline accurately calculates angular velocity, luminance, and IMU metrics, we will implement a suite of **Advanced Visualizations**. 

Unlike summary metrics (e.g., "Mean Fixation Duration"), these visualizations show the *structure* and *distribution* of behavior, which is critical for differentiating surgical skill levels.

## New Module: `src/analysis/visualizations.py`

We will create a dedicated module for generating these plots to keep the analysis logic clean.

### Visualization 1: The "Cognitive Fingerprint" (Fixation Duration)
**Type:** Violin Plot with strip overlay
**Goal:** Visualize the distribution of processing time.
**Hypothesis:** 
*   **Experts:** Bimodal distribution. A "hump" at ~150ms (visual scanning) and a distinct "hump" at ~400-600ms (manipulation/processing).
*   **Novices:** Unimodal, broad distribution (inefficient searching mixed with processing).

**Implementation Specs:**
*   **Input:** List of all fixation durations (ms) from the session (derived from I-VT classification).
*   **X-Axis:** Duration (ms). Log-scale often helps visualization.
*   **Style:** Seaborn `violinplot` (shows density) + `stripplot` (shows individual data points).
*   **Annotations:** Mark the mean and median.

### Visualization 2: The "Main Sequence" (Data Validation)
**Type:** Scatter Plot with Regression
**Goal:** Validate that the "angular velocity" data represents true biological eye movements.
**Logic:** Biological saccades follow a fixed relationship: as Amplitude ($A$) increases, Peak Velocity ($V_p$) increases logarithmically. 
$$ V_p \approx K \cdot (1 - e^{-A/C}) $$
**Implementation Specs:**
*   **Input:** List of all Saccade Amplitudes (degrees) and corresponding Peak Velocities (deg/s).
*   **X-Axis:** Amplitude (Degrees).
*   **Y-Axis:** Peak Velocity (Deg/s).
*   **Action:** Plot all saccades as semi-transparent dots. Overlay a fitted curve (or simple log-linear regression).
*   **Insight:** Outliers far above the curve represent tracker noise/artifacts. Points far below represent head movement compensating for eye movement.

### Visualization 3: The "Stress Test" (Temporal Dynamics)
**Type:** Stacked Time-Series
**Goal:** Correlate Cognitive Load with Motor Stability over time.
**Hypothesis:** High cognitive load (Pupil dilation) often precedes or coincides with motor instability (Head movement).

**Implementation Specs:**
*   **Input:** `final_gaze_data.csv` (Time, Luminance-Adjusted Pupil Residual, Gyroscope Magnitude).
*   **X-Axis:** Session Time (minutes).
*   **Subplot 1 (Top):** **Cognitive Load**.
    *   Line: Rolling mean (e.g., 5s window) of `pupil_residual`.
    *   Color: Shade regions where Residual > 1 Std Dev (High Load).
*   **Subplot 2 (Bottom):** **Motor Stability**.
    *   Line: Rolling mean of Total Gyro Magnitude ($\sqrt{x^2+y^2+z^2}$).
    *   Color: Shade regions of high instability.
*   **Insight:** Look for "Stress Events" where both lines spike simultaneously.

### Visualization 4: The "Stability Radar" (Directional Instability)
**Type:** Polar Histogram (Wind Rose)
**Goal:** Determine *how* the surgeon moves their head.
**Hypothesis:** 
*   **Experts:** Small, central cluster (locked head).
*   **Novices:** Large petals indicating specific movements (e.g., "nodding" pitch movements to change focus distance, or "yaw" movements to check instructions).

**Implementation Specs:**
*   **Input:** `head_gyro_x` (Pitch velocity) and `head_gyro_y` (Yaw velocity).
*   **Math:** 
    *   Magnitude $r = \sqrt{x^2 + y^2}$
    *   Angle $\theta = \arctan2(y, x)$
*   **Plot:** Polar bar chart (histogram of angles, weighted by magnitude).

---

## Execution Plan

### Step 1: Update `WholeSessionAnalyzer` to Return Raw Lists
Currently, `WholeSessionAnalyzer` returns summary stats (means/counts). We need to modify it to optionally return the **raw event lists** needed for plotting.

*   **Modify:** `src/analysis/whole_session_analysis.py`
*   **Action:** In `calculate_global_fixation_rate`, store the list of `fixation_durations` and `saccade_amplitudes`/`velocities` in the class instance so they can be accessed by the plotter.

### Step 2: Implement `src/analysis/visualizations.py`
Create the new class `GazeVisualizer`.
*   Function: `plot_fixation_distribution(fixation_durations, output_path)`
*   Function: `plot_main_sequence(amplitudes, velocities, output_path)`
*   Function: `plot_stress_timeline(timestamps, pupil_residuals, gyro_mags, output_path)`
*   Function: `plot_head_stability_radar(gyro_x, gyro_y, output_path)`

### Step 3: Integrate into Batch Process
*   **Modify:** `src/batch_process_with_heatmaps.py`
*   **Action:** Inside the loop (Step 4), after calculating metrics:
    1.  Instantiate `GazeVisualizer`.
    2.  Pass the raw data arrays from the analyzer to the visualizer.
    3.  Save plots to `reports/figures/[SubjectID]_viz_[type].png`.

---

## Expected Output
For each subject (e.g., `20231027T170020Z`), the following files will be generated in `reports/figures/`:

1.  `..._viz_cognitive_fingerprint.png` (Violin plot)
2.  `..._viz_main_sequence.png` (Scatter plot)
3.  `..._viz_stress_timeline.png` (Time series)
4.  `..._viz_stability_radar.png` (Polar plot)