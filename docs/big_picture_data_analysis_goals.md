## **Chicken Wing Dissection Project Big-Picture Data Analysis Goals**

Strategic Analysis Plan: Quantifying Surgical Skill via Oculometrics

### 1. Objective

This document outlines a data analysis plan to objectively quantify surgical skill by correlating eye movement dynamics with surgical performance. The primary performance metric is "Time to Completion," which is validated by the literature as a robust proxy for skill in the chicken wing training model (Jusue-Torres et al., 2013; Kaplan et al., 2015).

The core objective is to determine if **pure oculometric features**, derived from gaze data alone, can differentiate novice from expert surgeons. This analysis does not rely on semantic Areas of Interest (AOIs) at this stage.

### 2. Foundational Technical Step: Gaze Event Detection

To extract meaningful features, the raw, transformed gaze coordinates (`x, y, t`) must first be classified into a stream of oculomotor events.

- **Methodology:** A **Velocity-Threshold Identification (I-VT)** algorithm will be implemented. This is a standard, robust method for event classification that aligns with the literature (Hosp et al., 2021).
- **Process:**
    1. **Calculate Velocity:** Compute the angular velocity (in degrees per second) between consecutive gaze samples.
    2. **Classify Events:** Label each sample based on velocity thresholds:
        - **Fixation:** Low-velocity events (e.g., $< 30^\circ/s$), representing stable gaze.
        - **Saccade:** High-velocity ballistic movements (e.g., $> 300^\circ/s$), representing gaze shifts.
        - **Other:** Drifts, smooth pursuits, or noise.
- **Output:** The script will generate a new data file where every gaze sample is tagged with its event type (e.g., `gaze_state: "FIXATION"`). This classified data is the prerequisite for all subsequent feature extraction.

### 3. Phase 1 Data Analysis Goals

This analysis will focus on three key "Big-Picture" goals, with specific metrics derived from recent surgical eye-tracking literature.

### Goal A: The "Oculometric Efficiency" Index

- **Rationale:** Literature indicates that skill is defined by efficiency. Novice gaze is "chaotic" and covers a large area, while expert gaze is "structured and planned" (Hosp et al., 2021; Jusue-Torres et al., 2013).
- **Metrics & Hypotheses:**
    1. **Total Saccade Amplitude (Scanpath Length):** Hosp et al. (2021) identified this as a primary differentiator, finding novices (1956°) had over 4x the total saccadic amplitude of experts (481°).
        - **Hypothesis:** A strong positive correlation will exist between *Total Saccade Amplitude* and *Completion Time*.
    2. **Fixation Frequency (Rate):** Dalveren & Cagiltay (2020) found the *number of fixations* to be a key differentiator ($p<0.05$) between skill groups.
        - **Hypothesis:** A high `Fixations per Minute` rate will correlate positively with *Completion Time* (i.e., lower skill).

### Goal B: The "Motor Control & Planning" Signature

- **Rationale:** Beyond the *amount* of eye movement, the *consistency* of those movements reveals cognitive planning and motor control (Hosp et al., 2021).
- **Metric & Hypothesis:**
    1. **Saccade Peak Velocity (Standard Deviation):** Hosp et al. (2021) found that experts have more consistent, planned saccade speeds (lower StdDev: $93^\circ/s$) than intermediates ($121^\circ/s$).
        - **Hypothesis:** A lower standard deviation in peak saccade velocity will correlate with shorter *Completion Times* (i.e., higher skill).

### Goal C: The "Cognitive Stability" Signature

- **Rationale:** The high-precision, fine-motor nature of microsurgical dissection (specifically Tasks 3 & 4) suggests a "Quiet Eye" component. This is characterized by long, stable fixations on a target immediately preceding or during a critical motor action.
- **Metric & Hypothesis:**
    1. **Fixation Duration Distribution:**
        - **Hypothesis:** Experts will exhibit a bimodal distribution of fixation durations (a cluster of short "scanning" fixations and a distinct cluster of long "working" fixations). Novices are hypothesized to show a more unimodal, medium-length distribution, indicating continuous searching rather than discrete "work" phases.

## 4. Proposed Analysis Workflow

1. **Temporal Segmentation:** The raw data will be segmented using a `task_timestamps.csv` lookup file. This allows for a task-normalized comparison (e.g., comparing "Task 2: Intramuscular dissection" across all subjects), which is critical given the large variance in total completion times (3.5 min vs. 20+ min).
2. **Event Classification:** The I-VT algorithm (from Section 2) will be applied to the segmented, transformed gaze data for each subject.
3. **Feature Extraction:** The metrics from Goals A, B, and C will be computed for each subject (and for each of the “4 tasks”), resulting in a "Feature Matrix" (rows=subjects, columns=metrics).
4. **Statistical Analysis:** The Feature Matrix will be correlated (e.g., Pearson/Spearman) against "Time to Completion" to validate which gaze features are the strongest predictors of surgical skill. This will form the basis for subsequent classification modeling.