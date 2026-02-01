# Algorithms and Methodology

This document details the core algorithms used for gaze processing, perspective correction, and event detection.

## 1. Perspective Correction

To compare gaze data across different subjects or different sessions, we must map the gaze coordinates from the raw video frame (which moves as the user's head moves) to a fixed, static reference frame.

### ArUco Marker Tracking
We use **ArUco markers** (specifically from the `DICT_4X4_50` dictionary) placed at the corners of the surgical workspace.

*   **Detection**: We use OpenCV's `aruco.detectMarkers` to find markers in each video frame.
*   **Target Markers**: We look for specific IDs (default: 13, 14, 15, 16) that define the boundary of the region of interest.
*   **Corner Refinement**: We use sub-pixel refinement to get precise corner coordinates.

### Homography Transformation
Once the markers are detected, we calculate a **Homography Matrix (H)**.

1.  **Source Points**: The centers (or outer corners) of the detected ArUco markers in the current video frame.
2.  **Destination Points**: Fixed coordinates in the output image (e.g., [0,0], [1000,0], [1000,606], [0,606]).
3.  **Calculation**: `cv2.findHomography` computes the matrix that maps source points to destination points.
4.  **Application**:
    *   **Frame**: `cv2.warpPerspective` applies H to the video frame, "unwarping" it to a top-down view.
    *   **Gaze Point**: We apply the same matrix H to the raw gaze (x, y) coordinates to project them into the static workspace.

## 2. Saccade Detection

Saccades are rapid eye movements between fixations. Detecting them accurately is crucial for assessing surgical skill (e.g., expert search patterns vs. novice scanning).

### Primary: Adaptive MAD-based Detection
**File**: `src/analysis/adaptive_saccade_detector.py`

We use an adaptive algorithm based on the **Median Absolute Deviation (MAD)**, as proposed by Voloh et al. (2020). This is robust to noise and individual differences in eye movement dynamics.

**The Algorithm:**
1.  **Velocity Calculation**: We calculate angular velocity (deg/s) from 3D gaze direction vectors.
2.  **Iterative Thresholding**:
    *   Calculate the median velocity and MAD of the signal.
    *   `Threshold = Median + Lambda * 1.4826 * MAD` (where Lambda is typically 6).
    *   We iteratively refine this threshold by removing samples above the current threshold and re-calculating until it converges.
3.  **Physiological Filtering**:
    *   **Duration**: Events must be 20-100 ms.
    *   **Amplitude**: Events must be 0.5-50 degrees.
    *   **Velocity**: Peak velocity must be < 1000 deg/s.
4.  **Main Sequence Validation**: We verify that the relationship between amplitude and peak velocity follows the "Main Sequence" power law ($V_{peak} = k \cdot A^c$). Outliers are flagged.

### Legacy: I-VT (Velocity-Threshold Identification)
**File**: `src/processing/surgical_skill_analysis.py` (Class: `IVTEventClassifier`)

This is a simpler, standard algorithm often used as a baseline.

**The Algorithm:**
1.  **Fixed Thresholds**: It uses hard-coded thresholds (e.g., >300 deg/s for saccades, <30 deg/s for fixations).
2.  **Classification**:
    *   If velocity < 30 deg/s → **Fixation**
    *   If velocity > 300 deg/s → **Saccade**
    *   Otherwise → **Other** (smooth pursuit or noise)

*Note: The Adaptive MAD detector is preferred for this project as it accounts for individual physiological differences.*

## 3. Physics-Based Metrics

To ensure high-quality analysis, we calculate metrics based on physical principles rather than just pixels.

### Angular Velocity
Instead of calculating velocity from 2D pixel displacement (which is affected by head distance), we use the **3D Gaze Direction Vectors** provided by the eye tracker.

$$ \theta = \arccos(\vec{v}_{t} \cdot \vec{v}_{t-1}) $$
$$ \text{Velocity} = \frac{\theta}{\Delta t} $$

### Frame Luminance
We extract the mean grayscale intensity of the scene video. This is critical for the **Cognitive Load** analysis, where we must distinguish pupil dilation caused by mental effort from dilation caused by lighting changes (Pupillary Light Reflex).
