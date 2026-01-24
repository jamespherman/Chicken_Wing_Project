# ArUco Detection Debug and Fix Plan
**Date:** 2026-01-24

## Executive Summary

Investigation revealed that ArUco detection is **working correctly** for subjects with visible markers. The apparent "failure" was caused by:

1. Subject 20231012T122519Z does **not have visible ArUco markers** in the video
2. This subject was on the skip list in previous batch runs for this reason
3. When the batch processor ran today, it regenerated this subject's transformation_history.npy with 0 valid homographies
4. The whole-session analysis then used this invalid transformation history

**Evidence:**

| Subject | Transformation File Date | Valid Homographies | ArUco Markers Detected |
|---------|--------------------------|-------------------|------------------------|
| 20231012T122519Z | Jan 24, 2026 (regenerated) | 0 (0%) | NONE |
| 20231027T170020Z | Sep 1, 2025 (original) | 18,054 (97.7%) | 15, 16 |
| 20231027T171918Z | Sep 1, 2025 (original) | 28,210 (97.7%) | 13, 14, 15, 16 |

---

## Part 1: Root Cause Analysis

### 1.1 Why Subject 20231012T122519Z Has No Markers

The video for subject 20231012T122519Z appears to be from a session where:
- ArUco markers were not placed in the scene, OR
- The camera was positioned such that markers are not visible, OR
- A different marker dictionary was used

**Test Results:**
```
20231012T122519Z (10915 frames):
  Frame 2728: Markers found: NONE
  Frame 5457: Markers found: NONE
  Frame 8186: Markers found: NONE
```

### 1.2 Why Detection Works for Other Subjects

Subjects 20231027T170020Z and 20231027T171918Z have visible markers from the DICT_4X4_50 dictionary:
- Target markers: [13, 14, 15, 16]
- All four markers are present and detectable
- 97.7% detection rate indicates good marker visibility

### 1.3 The Confusion Source

The confusion arose because:
1. The batch processor ran `skip_existing: false`, which reprocessed ALL subjects
2. Subject 1's video was reprocessed despite having no markers
3. The new transformation_history.npy (with 0 valid homographies) overwrote any previous file
4. The whole-session analysis reported "0 valid transformations" for subject 1
5. This was misinterpreted as "ArUco detection is broken"

---

## Part 2: Debugging Plan

### Step 1: Verify Marker Presence in Videos

**Action:** Create a diagnostic script to thoroughly scan each video for ArUco markers.

```python
# debug_aruco_detection.py
import cv2
import numpy as np
from collections import Counter

def scan_video_for_markers(video_path, sample_interval=30):
    """Scan video at regular intervals to detect ArUco markers."""
    cap = cv2.VideoCapture(video_path)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    except AttributeError:
        detector = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    marker_counts = Counter()
    frames_with_markers = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if detector:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

            if ids is not None:
                frames_with_markers += 1
                for marker_id in ids.flatten():
                    marker_counts[marker_id] += 1

        frame_idx += 1

    cap.release()

    return {
        'total_frames': total_frames,
        'sampled_frames': total_frames // sample_interval,
        'frames_with_markers': frames_with_markers,
        'marker_counts': dict(marker_counts),
        'detection_rate': frames_with_markers / (total_frames // sample_interval) * 100
    }
```

**Expected Output:** Report showing which markers are present in each video.

### Step 2: Test Different Marker Dictionaries

**Action:** If subject 1 has markers but from a different dictionary, test other dictionaries.

```python
DICTIONARIES_TO_TEST = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_ARUCO_ORIGINAL,
]
```

### Step 3: Enhance Detection Parameters

**Action:** If markers are present but detection is failing, tune parameters:

```python
def get_enhanced_parameters():
    params = cv2.aruco.DetectorParameters()

    # Adaptive thresholding
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 80
    params.adaptiveThreshWinSizeStep = 5

    # Marker filtering
    params.minMarkerPerimeterRate = 0.01  # More permissive
    params.maxMarkerPerimeterRate = 4.0   # More permissive

    # Corner refinement
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 100

    return params
```

### Step 4: Add Robust Error Handling

**Action:** Update `gaze_on_perspective_corrected_frames_refactored.py` to:

1. Log when markers are not detected
2. Log which specific markers are missing
3. Track detection statistics per-chunk
4. Provide meaningful warnings in output

```python
# Add to _process_frame_chunk()
if ids is None:
    logger.debug(f"Frame {frame_index}: No markers detected")
else:
    detected_ids = set(ids.flatten())
    missing = set(target_markers) - detected_ids
    if missing:
        logger.debug(f"Frame {frame_index}: Missing markers {missing}")
```

### Step 5: Update Configuration

**Action:** Add subject 20231012T122519Z to skip list in config.json:

```json
{
    "subjects_to_skip": ["20231012T122519Z"],
    ...
}
```

---

## Part 3: Fix Implementation Plan

### Fix 1: Restore Original Transformation History

**Priority:** HIGH
**Action:** The transformation_history.npy files for subjects 2 and 3 are still valid (from Sep 2025). Only subject 1 was overwritten.

Since subject 1 has no markers, its transformation_history should remain empty OR we should:
1. Add it to the skip list
2. Mark it as "no marker calibration available"
3. Process it with physics-only metrics (no gaze position transformation)

### Fix 2: Add Detection Diagnostics to Batch Processor

**Priority:** MEDIUM
**Action:** Before processing, run a quick marker scan:

```python
def check_marker_availability(video_path, target_markers, sample_size=10):
    """Quick check if target markers are present in video."""
    # Sample 10 frames evenly distributed
    # Return True if all target markers found at least once
    pass

# In process_single_subject():
if not check_marker_availability(video_path, target_markers):
    logger.warning(f"Subject {subject}: Target markers not found in video")
    # Optionally add to skip list or mark for physics-only processing
```

### Fix 3: Graceful Degradation for Missing Markers

**Priority:** MEDIUM
**Action:** When markers are not available, still extract physics-based data:

This is **already implemented** as of today's changes:
- `create_final_csv_refactored.py` now extracts angular velocity, pupil, IMU even without valid homographies
- Whole-session analysis works with physics data alone

### Fix 4: Add Pre-Processing Validation

**Priority:** LOW
**Action:** Add a validation step before batch processing:

```python
def validate_subject_data(subject_folder, config):
    """Validate that subject has required data for processing."""
    checks = {
        'has_video': (subject_folder / config['video_filename']).exists(),
        'has_gaze': (subject_folder / config['gaze_filename']).exists(),
        'has_imu': (subject_folder / 'imudata.gz').exists(),
        'has_markers': check_marker_availability(...)
    }
    return checks
```

---

## Part 4: Recommended Actions

### Immediate Actions

1. **Update config.json** to add subject 20231012T122519Z to skip list (if marker-based processing is required)

2. **Document subject characteristics:**
   - Subject 1: No ArUco markers, physics-only analysis available
   - Subjects 2 & 3: Full marker-based gaze position + physics analysis

3. **Do NOT regenerate transformation_history.npy** for subjects 2 and 3 - they have valid data

### Future Enhancements

1. **Add marker scanning tool** for new subjects before processing
2. **Implement adaptive processing** that automatically falls back to physics-only mode
3. **Add detection quality metrics** to processing logs
4. **Consider alternative calibration** methods for subjects without markers

---

## Part 5: Verification Checklist

After implementing fixes, verify:

- [ ] Subject 20231012T122519Z is on skip list OR processed with physics-only mode
- [ ] Subjects 20231027T170020Z and 20231027T171918Z retain valid transformation histories
- [ ] Batch processor logs detection statistics
- [ ] Whole-session analysis correctly handles subjects without gaze position data
- [ ] New subjects are validated before processing

---

## Appendix: Current ArUco Configuration

**Marker Dictionary:** DICT_4X4_50
**Target Markers:** [13, 14, 15, 16]
**Output Canvas:** 1000 x 606 pixels
**Detection Mode:** cv2.aruco.ArucoDetector (OpenCV 4.7+)

**Parameters (default):**
- adaptiveThreshWinSizeMin: 3
- adaptiveThreshWinSizeMax: 23
- adaptiveThreshWinSizeStep: 10
- cornerRefinementMethod: CORNER_REFINE_NONE

**Parameters (enhanced, when enabled):**
- adaptiveThreshWinSizeMin: 3
- adaptiveThreshWinSizeMax: 80
- adaptiveThreshWinSizeStep: 5
- cornerRefinementMethod: CORNER_REFINE_SUBPIX
