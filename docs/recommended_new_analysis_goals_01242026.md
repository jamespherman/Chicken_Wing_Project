# Recommended New Analysis Goals (01/24/2026)

Based on the analysis of the "Big-Picture Data Analysis Goals" and the available data in the codebase (specifically `gazedata.gz`), the following additional analysis goals are recommended to provide a more comprehensive assessment of surgical skill.

## Goal D: Cognitive Load via Pupil Diameter

**Rationale:**
Pupil dilation is a well-established physiological proxy for cognitive load and mental effort. In surgical training, higher cognitive load (often seen in novices or during complex tasks) is correlated with increased pupil diameter. Experts generally exhibit lower, more stable cognitive load due to automaticity.

**Data Availability:**
The raw data files (`gazedata.gz`) contain `pupildiameter` fields for both left and right eyes.

**Implementation Steps:**
1.  **Update Processing:** Modify `src/processing/create_final_csv_refactored.py` to extract `pupildiameter` (averaging left and right eye if both are valid) and include it in `final_gaze_data.csv`.
2.  **Metric Calculation:**
    - **Mean Pupil Diameter:** Baseline-corrected mean diameter during tasks.
    - **Pupil Diameter Variance:** Standard deviation of diameter (indicating fluctuations in load).
    - **Index of Cognitive Activity (ICA):** (Optional/Advanced) Frequency of rapid small dilations.
3.  **Hypothesis:** Novices will show higher mean pupil diameter and higher variance compared to experts.

## Goal E: Motor Stability via Head Posture

**Rationale:**
Microsurgery (like the chicken wing model) requires extreme physical stability, including head posture. Frequent head movements may indicate a lack of stability or difficulty in maintaining a visual fix.

**Data Availability:**
The raw data files (`gazedata.gz`) contain `gazeorigin` (3D coordinates of the eye position) for both eyes. This can serve as a robust proxy for head position relative to the eye tracker.

**Implementation Steps:**
1.  **Update Processing:** Modify `src/processing/create_final_csv_refactored.py` to extract `gazeorigin` coordinates.
2.  **Metric Calculation:**
    - **Head Movement Amplitude:** Total path length of the head (gaze origin) during the task.
    - **Head Stability Index:** Variance of the head position in 3D space.
3.  **Hypothesis:** Experts will demonstrate significantly lower head movement amplitude and higher stability than novices.

## Goal F: Spatial Strategy via Gaze Entropy

**Rationale:**
While "Goal A" looks at total scanpath length, it doesn't capture the *randomness* or *predictability* of the search strategy. Novices often exhibit "random search," while experts have "structured search." Entropy measures can quantify this without needing semantic AOIs (which are time-consuming to annotate).

**Data Availability:**
Derived from existing `(x, y)` gaze coordinates.

**Implementation Steps:**
1.  **Grid Discretization:** Divide the visual field into a grid (e.g., 5x5 or 10x10).
2.  **Transition Matrix:** Calculate the probability of moving from cell $i$ to cell $j$.
3.  **Metric Calculation:**
    - **Stationary Entropy:** Randomness of where they look (spatial distribution).
    - **Transition Entropy:** Randomness of *how* they move between locations (scanpath predictability).
4.  **Hypothesis:** Novices will have higher transition entropy (more random movements), while experts will have lower entropy (more predictable, schematic search patterns).
