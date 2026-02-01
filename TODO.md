# To-Do

This file lists potential optimizations, improvements, and planned features for the Chicken Wing Surgical Cognition Analysis project.

**Last Updated:** 2026-02-01

---

## Completed

### Saccade Detection (2026-01-31)
- [x] **Implement MAD-based adaptive saccade detection** - Replaced fixed 300 deg/s threshold with iterative MAD algorithm
- [x] **Calculate saccade amplitude from gaze direction vectors** - Using arccos(dot product) instead of velocity integration
- [x] **Add physiological filtering** - Duration 20-100ms, amplitude 0.5-50°, velocity <1000 deg/s
- [x] **Main sequence validation** - Flag saccades >3 SD from power-law fit

### Whole-Session Analysis (2026-01-31)
- [x] **Implement Goal A: Oculometric Efficiency** - Fixation rate, duration, saccade metrics
- [x] **Implement Goal B: Cognitive Load** - Pupil-luminance regression
- [x] **Implement Goal C: Motor Stability** - Integrated gyroscopic motion
- [x] **Create clinical visualizations** - Cognitive fingerprint, main sequence, stress timeline, stability radar

### Cohort Analysis (2026-01-31)
- [x] **Cross-subject aggregation** - Load all subject JSON files into unified DataFrame
- [x] **Cohort statistics** - Mean, std, median, quartiles for all metrics
- [x] **Outlier detection** - IQR-based identification
- [x] **CSV export** - For external statistical analysis

### Pupil-Luminance Temporal Kernel (2026-02-01)
- [x] **Implement Erlang gamma kernel** - h(t) = (t/t_max)^n × exp(n × (1 - t/t_max))
- [x] **Per-subject kernel fitting** - L-BFGS-B optimization with physiological bounds
- [x] **Causal convolution** - Only past luminance affects current pupil
- [x] **Kernel fit visualization** - 4-panel plot showing IRF, time series, scatter plots
- [x] **New metrics** - kernel_t_max_ms, kernel_n, regression_r_squared_convolved

### Duration Regression Analysis (2026-02-01)
- [x] **Lasso/Ridge regression** - Predict recording duration from oculometric metrics
- [x] **Exclude tautological predictors** - saccade_count, fixation_count, etc.
- [x] **Outlier analysis** - Identified 22-second recording as leverage point
- [x] **Comprehensive visualization** - 12-panel regression figure
- [x] **Incorporate kernel metrics** - regression_r_squared_convolved selected by Lasso

---

## High Priority

### Pupil-Luminance Improvement (Critical)
- [ ] **Implement gaze-contingent luminance extraction** - Extract luminance at point of gaze, not mean frame
  - Create `src/processing/gaze_luminance.py`
  - Add `gaze_luminance` column to final CSV
  - Expected R² improvement from 0.007 to 0.05-0.20
  - See: `docs/2026-02-01_pupil_luminance_next_steps.md`

- [ ] **Validate gaze-contingent luminance on single subject** - Before batch processing

- [ ] **Re-run cohort analysis with corrected luminance** - Update all derived metrics

### Code Quality
- [ ] **Improve test coverage** - Add unit tests for:
  - `src/processing/create_final_csv_refactored.py`
  - `src/processing/gaze_on_perspective_corrected_frames_refactored.py`
  - `src/analysis/pupil_luminance_kernel.py`
  - `src/analysis/adaptive_saccade_detector.py`

---

## Medium Priority

### Configuration & Flexibility
- [ ] **Externalize frame dimensions** - Move hardcoded `frame_width`/`frame_height` to `config.json`
- [ ] **Parameterize data filenames** - Move `scenevideo.mp4`, `gazedata.gz` to `config.json`

### Advanced Pupil Modeling (Phase 2)
- [ ] **Extract pre-trained video transformer features** - Use VideoMAE/TimeSformer as frozen backbone
- [ ] **Gaze-weighted feature pooling** - Pool spatial features at gaze location
- [ ] **Train temporal model** - Features → pupil prediction
- [ ] See: `docs/2026-02-01_pupil_luminance_next_steps.md`

### Analysis Enhancements
- [ ] **Add cross-validation to duration regression** - Currently only in-sample R²
- [ ] **Implement bootstrap confidence intervals** - For cohort statistics
- [ ] **Add effect size calculations** - Cohen's d for group comparisons (when groups available)

---

## Low Priority

### Infrastructure
- [ ] **Add log rotation/cleanup** - Manage accumulating log files in `reports/logs`
- [ ] **Update README.md** - Configuration section incorrectly references main() function

### Future Research Directions (Phase 3)
- [ ] **Custom gaze-guided transformer** - End-to-end video → pupil model
- [ ] **Requires more data** - Current 14 subjects insufficient for training
- [ ] See: `reports/logs/transformer_pupil_research_report.md`

### Documentation
- [ ] **API documentation** - Docstrings for all public functions
- [ ] **User guide** - Step-by-step processing instructions
- [ ] **Architecture diagram** - Visual overview of data flow

---

## Known Issues

### Pupil-Luminance Regression
- **Issue:** Mean frame luminance is a poor predictor (median R² = 0.007)
- **Cause:** Pupil responds to foveal illumination, not whole-scene brightness
- **Solution:** Implement gaze-contingent luminance (HIGH PRIORITY)

### Duration Correlations
- **Issue:** Correlations with duration are driven by 2 outlier subjects
- **Example:** regression_r_squared_convolved correlation inverts when outliers removed (r=+0.53 → r=-0.57)
- **Mitigation:** Report results with and without outliers; increase sample size

### Kernel Fitting Failures
- **Issue:** 3 of 14 subjects fall back to canonical kernel parameters
- **Cause:** Very weak luminance-pupil relationship in those subjects
- **Mitigation:** Will likely improve with gaze-contingent luminance

---

## File Locations

### Planning Documents
- `docs/2026-02-01_pupil_luminance_next_steps.md` - Detailed implementation plan for luminance improvement
- `docs/2026-01-31_aggregation_implementation_plan.md` - Cohort analysis plan
- `docs/2026-01-24_visualization_implementation_plan.md` - Visualization plan

### Analysis Reports
- `reports/logs/transformer_pupil_research_report.md` - Research on transformer-based approaches
- `reports/logs/duration_regression_kernel_report.txt` - Duration regression with kernel metrics
- `reports/logs/cohort_analysis_report.txt` - Cross-subject statistics

### Key Source Files
- `src/analysis/pupil_luminance_kernel.py` - Temporal kernel implementation
- `src/analysis/adaptive_saccade_detector.py` - MAD-based saccade detection
- `src/analysis/whole_session_analysis.py` - Main analysis module
- `src/analysis/cohort_analyzer.py` - Cross-subject aggregation
- `src/analysis/visualizations.py` - Clinical visualizations
