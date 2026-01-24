# 2026-24-01 Development Implementation Plan

## Overview
This document outlines the detailed implementation plan to execute the revised analysis roadmap, transitioning from pixel-based to physics-based (degrees) oculometric analysis with whole-session global metrics.

---

## Phase 1: Physics Update Implementation

### Task 1.1: Angular Velocity Calculation
**File to modify:** `src/processing/create_final_csv_refactored.py`

**Implementation Steps:**
1. Modify `load_gaze_data_stream()` to extract `gazedirection` vectors from both eyes
2. Create function `calculate_angular_velocity(gaze_direction_t, gaze_direction_t_minus_1, delta_t)`:
   - Compute dot product of consecutive 3D unit vectors
   - Calculate angular change: `theta = arccos(v_t . v_{t-1})`
   - Compute angular velocity: `omega = theta / delta_t` (convert to deg/s)
3. Handle edge cases: missing data, first sample, invalid vectors
4. Add `angular_velocity_deg_s` column to output CSV
5. Also extract and output `pupil_diameter_left`, `pupil_diameter_right`, `pupil_diameter_avg`

### Task 1.2: Luminance Extraction
**File to modify:** `src/processing/create_final_csv_refactored.py`

**Implementation Steps:**
1. Create function `calculate_frame_luminance(frame)`:
   - Convert frame to grayscale
   - Calculate mean pixel intensity (0-255)
2. Integrate with video processing loop (or load from video separately)
3. Create a frame luminance lookup dictionary indexed by frame number
4. Merge luminance values with gaze timestamps via nearest frame matching
5. Add `frame_luminance` column to output CSV

### Task 1.3: IMU Data Integration
**File to modify:** `src/processing/create_final_csv_refactored.py`

**Implementation Steps:**
1. Create function `load_imu_data_stream(imu_path)`:
   - Read compressed `imudata.gz` file
   - Parse gyroscope and accelerometer data with timestamps
2. Create function `sync_imu_to_gaze(gaze_timestamp, imu_data)`:
   - Find nearest IMU sample for each gaze timestamp
   - Use interpolation if timestamps don't align exactly
3. Extract gyroscope X, Y, Z values
4. Add columns: `head_gyro_x`, `head_gyro_y`, `head_gyro_z`

### Task 1.4: Refactor create_final_csv_refactored.py
**Consolidate all Phase 1 changes:**
- Update main processing function to handle all new data streams
- Ensure efficient memory usage with streaming approach
- Update CSV schema with all new columns
- Add validation and error handling for new data types

---

## Phase 2: Analysis Module Updates

### Task 2.1: Create Whole-Session Analysis Module
**New file:** `src/analysis/whole_session_analysis.py`

**Implementation Steps:**
1. Create class `WholeSessionAnalyzer`:
   - Load enhanced final_gaze_data.csv with all new columns
   - Implement Goal A-D metric calculations

2. **Goal A: Oculometric Efficiency**
   - Function `calculate_global_fixation_rate(angular_velocities, timestamps)`:
     - Apply I-VT threshold on angular velocity (30 deg/s)
     - Count fixation events
     - Return fixations per second (Hz)

3. **Goal B: Cognitive Load (Pupil Response)**
   - Function `calculate_luminance_adjusted_pupil_residuals(pupil_diameters, luminances)`:
     - Fit linear regression: Diameter ~ Luminance
     - Calculate residuals
     - Return mean residual and residual time series

4. **Goal C: Motor Stability**
   - Function `calculate_integrated_gyro_motion(gyro_x, gyro_y, gyro_z, timestamps)`:
     - Sum absolute values of angular velocity over time
     - Return total integrated motion

5. **Goal D: Visual Strategy**
   - Function `classify_gaze_target(video_path, gaze_coords, timestamps)`:
     - Extract ROI around gaze point from video frame
     - Convert to HSV color space
     - Apply thresholds: Grey/Silver = Tool, Pink/Yellow = Tissue
     - Return percentage time on Tool vs Tissue

### Task 2.2: Update SurgicalSkillAnalyzer
**File to modify:** `src/processing/surgical_skill_analysis.py`

**Implementation Steps:**
1. Update `IVTEventClassifier` to use `angular_velocity_deg_s` directly:
   - Add parameter `use_precomputed_velocity=True`
   - Skip internal velocity calculation when precomputed
   - Ensure threshold is in degrees/second (30 deg/s)

2. Update `OculometricFeatureExtractor`:
   - Add method for extracting luminance-corrected pupil metrics
   - Add method for IMU-based head stability metrics
   - Add method for visual strategy metrics

3. Update `SurgicalSkillAnalyzer.analyze_subject_task()`:
   - Include new metrics in feature extraction
   - Handle whole-session mode (no task segmentation needed)

---

## Phase 3: Batch Processing and Reporting

### Task 3.1: Update Batch Processor
**File to modify:** `src/batch_process_with_heatmaps.py`

**Implementation Steps:**
1. Add Step 4: Whole-Session Analysis
   - Call `WholeSessionAnalyzer` after CSV generation
   - Generate subject-level metrics summary
2. Update configuration to enable/disable new analysis features
3. Add new output paths for analysis results

### Task 3.2: Run Batch Processing
**Execute on all 3 subjects:**
- 20231012T122519Z
- 20231027T170020Z
- 20231027T171918Z

**Validation:**
- Verify all new columns present in output CSVs
- Check for reasonable value ranges
- Compare metrics across subjects

### Task 3.3: Generate Correlation Report
**Implementation:**
1. Collect metrics from all subjects
2. Use recording duration as performance proxy
3. Calculate Pearson and Spearman correlations
4. Generate visualization of metrics by subject
5. Output summary report

---

## Implementation Order and Dependencies

```
Phase 1: Physics Update (Data Layer)
├── Task 1.1: Angular Velocity ──┐
├── Task 1.2: Luminance ─────────┼──→ Task 1.4: Consolidate create_final_csv
└── Task 1.3: IMU Integration ───┘

Phase 2: Analysis Updates (Analysis Layer)
├── Task 2.1: WholeSessionAnalyzer (depends on Phase 1)
└── Task 2.2: SurgicalSkillAnalyzer updates (depends on Task 2.1)

Phase 3: Execution (Integration Layer)
├── Task 3.1: Batch Processor update (depends on Phase 2)
├── Task 3.2: Run batch (depends on Task 3.1)
└── Task 3.3: Generate report (depends on Task 3.2)
```

---

## New Output CSV Schema

After implementation, `[SUBJECT]_final_gaze_data.csv` will contain:

| Column | Type | Description |
|--------|------|-------------|
| gaze_timestamp | float | Time in seconds |
| transformed_gaze_x | float | X coordinate in workspace pixels |
| transformed_gaze_y | float | Y coordinate in workspace pixels |
| active_frame_index | int | Video frame number |
| active_frame_time | float | Video frame timestamp |
| angular_velocity_deg_s | float | Eye angular velocity (deg/s) |
| pupil_diameter_left | float | Left pupil diameter (mm) |
| pupil_diameter_right | float | Right pupil diameter (mm) |
| pupil_diameter_avg | float | Average pupil diameter (mm) |
| frame_luminance | float | Frame brightness (0-255) |
| head_gyro_x | float | Head rotation rate X (deg/s) |
| head_gyro_y | float | Head rotation rate Y (deg/s) |
| head_gyro_z | float | Head rotation rate Z (deg/s) |

---

## Expected Outputs

1. **Enhanced CSVs** for each subject with physics-based metrics
2. **Whole-session analysis results** per subject:
   - Global fixation rate (Hz)
   - Mean pupil residual (cognitive load indicator)
   - Integrated gyro motion (head stability score)
   - Tool/Tissue gaze ratio
3. **Cross-subject comparison report** with correlations
4. **Visualizations** of new metrics

---

## Success Criteria

- [ ] All gaze samples have valid angular velocity calculations
- [ ] Luminance values extracted for all frames
- [ ] IMU data successfully synchronized with gaze timestamps
- [ ] Whole-session metrics computed for all 3 subjects
- [ ] Correlation analysis completed with recording duration
- [ ] Final report generated with findings
