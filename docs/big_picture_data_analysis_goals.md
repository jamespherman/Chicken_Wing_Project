# Chicken Wing Dissection Project: Big-Picture Data Analysis Goals
**Revision Date:** January 24, 2026

## 1. Objective

This document outlines the strategic analysis plan to objectively quantify surgical skill by correlating eye movement dynamics, head stability, and visual strategy with surgical performance. 

The primary analysis constraint is to derive **pure oculometric and sensor-based features** that differentiate novice from expert surgeons. While "Time to Completion" remains the gold-standard proxy for skill, this analysis seeks to identify the *cognitive and motor underpinnings* of that speed.

## 2. Foundational Technical Step: Physics-Based Event Detection

To ensure scientific rigor, we move beyond pixel-based analysis to **physics-based metrics**.

-   **Data Source:** Tobii Pro Glasses 3 `gazedata.gz` (specifically 3D `gazedirection` vectors).
-   **Methodology:** **Velocity-Threshold Identification (I-VT)** using **Angular Velocity**.
-   **Process:**
    1.  **3D Vector Math:** Calculate the angular change $\theta$ between consecutive 3D gaze direction vectors: $\theta = \arccos(\vec{v}_t \cdot \vec{v}_{t-1})$.
    2.  **Angular Velocity:** Compute degrees per second ($\omega = \theta / \Delta t$).
    3.  **Classify Events:**
        -   **Fixation:** Stable gaze on a specific 3D coordinate (Low angular velocity).
        -   **Saccade:** Ballistic eye movement between targets (High angular velocity).

## 3. Analysis Goals

### Goal A: The "Oculometric Efficiency" Index
*Rationale: Novice gaze is "chaotic" and searching; expert gaze is "structured" and knowing.*

1.  **Global Fixation Rate (Hz):**
    *   **Hypothesis:** Novices exhibit a higher frequency of fixations per minute (searching behavior) compared to experts (efficient information gathering).
2.  **Total Scanpath Length (Degrees):**
    *   **Hypothesis:** Novices cover a vastly larger total angular distance than experts to acquire the same visual information.

### Goal B: The "Motor Control & Planning" Signature
*Rationale: Consistency of movement reveals cognitive confidence.*

1.  **Saccade Peak Velocity Consistency:**
    *   **Hypothesis:** Experts demonstrate a lower standard deviation in peak saccade velocity, indicating pre-planned, ballistic movements. Novices show higher variance due to mid-flight corrections and uncertainty.

### Goal C: The "Cognitive Load" Monitor (Pupillometry)
*Rationale: Pupil dilation proxies cognitive effort, but is confounded by light reflex.*

1.  **Luminance-Corrected Pupil Residuals:**
    *   **Method:** Regress pupil diameter against scene luminance (extracted from video frames).
    *   **Metric:** The *residual* (Observed - Predicted). Positive residuals indicate dilation driven by cognitive load rather than darkness.
    *   **Hypothesis:** Novices will show significantly higher positive residuals (higher cognitive load) than experts.

### Goal D: The "Physical Stability" Index (IMU)
*Rationale: Microsurgery requires locking the head and upper body to stabilize the hands.*

1.  **Integrated Head Gyroscopy:**
    *   **Method:** Integrate the magnitude of the gyroscope data (deg/s) from `imudata.gz` over time.
    *   **Hypothesis:** Experts will minimize head rotation (lower integrated score); novices will move their head frequently to adjust viewing angles.

### Goal E: The "Visual Strategy" Class (Tool vs. Tissue)
*Rationale: Experts look where they are *going* (tissue); novices look at what they are *holding* (tool).*

1.  **Tool-Gaze Ratio:**
    *   **Method:** Color-thresholding (HSV) at the gaze coordinate in the scene video to classify "Silver/Grey" (Tool) vs. "Pink/Yellow" (Tissue).
    *   **Hypothesis:** A higher ratio of time spent looking at the Tool correlates with lower skill levels.

## 4. Execution Workflow

1.  **Whole-Session Analysis:** Due to the variability in task segmentation availability, the primary analysis will treat the **entire recording** as a single performance block.
2.  **Feature Extraction:** The `create_final_csv` pipeline will calculate frame-by-frame metrics (Angular Velocity, Luminance, Head Gyro, Target Class).
3.  **Statistical Correlation:** The resulting Feature Matrix (one row per subject) will be correlated against **Total Duration** (a proxy for skill).