# 2026-01-24_revised_analysis_roadmap.md

## Project Status Update
We are shifting the immediate focus from task-segmented analysis (due to pending timestamps) to a "Whole-Session" Global Analysis. We are also introducing a rigorous "Physics Update" to ensure all oculometric calculations are performed in Degrees rather than Pixels, utilizing the Tobii Pro Glasses 3 3D vector data.

## Phase 1: The "Physics Update" (Data Integrity)
Before running any I-VT algorithms, we must ensure the underlying data represents physiological eye movements (degrees), not screen pixels.

### 1. Angular Velocity Calculation (The "Degrees" Fix)
**Rationale:** The developer guide (p.5, p.50) confirms data is in normalized coordinates or mm. I-VT requires angular velocity.
**Implementation:**
*   Modify `src/processing/create_final_csv_refactored.py`.
*   Parse `gazedirection` (3D unit vector) from `gazedata.gz`.
*   Calculate instantaneous angular change between frame $t$ and $t-1$:
    $$ \theta = \arccos( \vec{v}_t \cdot \vec{v}_{t-1} ) $$
*   Calculate Angular Velocity: $\omega = \theta / \Delta t$.
*   **Output:** Add `angular_velocity_deg_s` column to `final_gaze_data.csv`.

### 2. Luminance Extraction (for Pupil Correction)
**Rationale:** Pupil diameter changes with light. We need to regress this out.
**Implementation:**
*   Modify `src/processing/create_final_csv_refactored.py`.
*   During the video processing loop, calculate the mean grayscale intensity (0-255) of the current frame.
*   **Output:** Add `frame_luminance` column to `final_gaze_data.csv`.

### 3. IMU Processing (Head Stability)
**Rationale:** Using `imudata.gz` allows us to measure motor stability directly (Dev Guide p.44).
**Implementation:**
*   Modify `src/processing/create_final_csv_refactored.py` to read `imudata.gz`.
*   Sync IMU timestamps with Gaze timestamps.
*   Extract Gyroscope X, Y, Z data.
*   **Output:** Add `head_gyro_x`, `head_gyro_y`, `head_gyro_z` columns.

---

## Phase 2: The "Whole-Session" Analysis Goals

Since we lack start/stop timestamps for specific tasks, we will analyze the **entire recording** as a single performance block.

### Goal A: Oculometric Efficiency (Global)
*   **Metric:** **Global Fixation Rate (Hz)**.
*   **Logic:** Does the surgeon fixate frequently (novice searching) or stably (expert planning)?
*   **Input:** `angular_velocity_deg_s` (from Phase 1).

### Goal B: Cognitive Load (Pupil Response)
*   **Metric:** **Luminance-Adjusted Pupil Residuals**.
*   **Logic:**
    1.  Perform Linear Regression: $Diameter \sim \beta \times Luminance$.
    2.  Calculate Residuals: $R = Diameter_{observed} - Diameter_{predicted}$.
    3.  Mean $R > 0$ indicates high cognitive load.

### Goal C: Motor Stability (Head)
*   **Metric:** **Integrated Gyroscopic Motion**.
*   **Logic:** Sum of absolute rotational velocity over time. Lower is better (steadier head).
*   **Input:** IMU Gyro columns.

### Goal D: Visual Strategy (Tool vs. Tissue)
*   **Metric:** **Tool-Gaze Percentage**.
*   **Logic:** Novices look at tools; experts look at tissue.
*   **Implementation:**
    *   Use OpenCV on `scenevideo.mp4` at the gaze coordinate.
    *   Convert ROI to HSV.
    *   Simple threshold: Grey/Silver (Tool) vs. Pink/Yellow (Tissue).
    *   Calculate ratio of time spent on Tool vs. Tissue.

---

## Phase 3: Execution Plan

1.  **Refactor `create_final_csv`:** This is the heavy lifting. It needs to ingest IMU data, calculate 3D angles, and compute frame luminance.
2.  **Refactor `SurgicalSkillAnalyzer`:** Update it to use `angular_velocity_deg_s` directly for I-VT, instead of calculating it from x/y coordinates.
3.  **Run Batch:** Process the 3 existing subjects.
4.  **Generate Report:** Correlate these new metrics against the *Total Duration* of the recording (as a proxy for performance, since better surgeons likely finish faster).