# Analysis Goals and Metrics

This project quantifies surgical skill using four key dimensions ("Goals"). These metrics are derived from the raw gaze and kinematic data.

## Goal A: Oculometric Efficiency
**Concept**: Experts search less and recognize more. Novices scan broadly and erratically.

**Metrics:**
*   **Fixation Rate (Hz)**: Number of fixations per second. High rates often indicate inefficient visual search or confusion.
*   **Scanpath Length**: The total angular distance covered by the eye. Experts typically have shorter, more efficient scanpaths.
*   **Saccade/Fixation Ratio**: The proportion of time spent searching vs. processing.

## Goal B: Cognitive Load
**Concept**: Mental effort manifests in physiological responses. High cognitive load (stress, difficult task) causes pupil dilation.

**Methodology:**
1.  **Pupillary Light Reflex (PLR) Correction**: Lighting changes cause pupil changes. We use a **Temporal Kernel Regression** model to predict the pupil response solely due to lighting (using `frame_luminance`).
2.  **Residual Analysis**: We subtract the predicted PLR from the actual pupil diameter.
    *   `Residual = Actual_Pupil - Predicted_PLR_Pupil`
3.  **Interpretation**: Positive residuals indicate cognitive dilation—mental effort required beyond what the lighting environment dictates.

## Goal C: Motor Stability
**Concept**: A steady head indicates control and confidence.

**Metrics:**
*   **Integrated Gyroscopic Motion**: We integrate the absolute rotation rates from the IMU (Head Gyroscope) over time.
    *   $\int (|\omega_x| + |\omega_y| + |\omega_z|) dt$
*   **Interpretation**: Lower values indicate a stable head posture, characteristic of expert surgeons. High values indicate excessive head movement (searching, adjusting view).

## Goal D: Visual Strategy
**Concept**: Experts look at the tissue they are operating on. Novices often look at their tools.

**Methodology:**
*   **Target Classification**: We analyze the image region around the gaze point.
*   **HSV Color Filtering**:
    *   **Tools**: Typically metallic/grey (Low Saturation).
    *   **Tissue**: Pink/Red/Yellow (Specific Hues, High Saturation).
*   **Metric**: `Tool/Tissue Ratio`. A lower ratio (more attention on tissue) correlates with higher skill.

## Visualizations

### 1. Gaze Heatmaps
**File**: `src/analysis/gaze_heatmap_analysis.py`

*   **Heatmap**: A 2D density plot of gaze points on the stabilized frame. Hotter colors = more attention.
*   **Scatter Plot**: Shows raw points, colored by time (purple=start, yellow=end) to visualize the surgical trajectory.
*   **Contour Map**: Topographic view of attention density.

### 2. Clinical Dashboards
**File**: `src/analysis/cohort_visualizations.py`

*   **Cognitive Fingerprint (Radar Chart)**: A multi-axis chart comparing a subject against the "Ideal Expert" profile across all goals.
*   **Stress Timeline**: A plot of Pupil Residuals (Cognitive Load) over time, annotated with key surgical events.
*   **Main Sequence**: A scatter plot of Saccade Amplitude vs. Peak Velocity to verify data quality and physiological normality.
