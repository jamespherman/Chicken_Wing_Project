# Pupil-Luminance Modeling: Next Steps Implementation Plan

**Date:** 2026-02-01
**Status:** Planning
**Priority:** High

---

## Executive Summary

This document outlines the recommended path forward for improving the pupil-luminance regression analysis in the surgical cognition pipeline. The current approach using mean frame luminance achieves very poor predictive performance (median R² = 0.007), and we have identified that this is fundamentally due to using the wrong luminance measure. This plan details a staged approach from simple improvements to advanced transformer-based methods.

---

## Current State Assessment

### What We Have Implemented

1. **Temporal Kernel Convolution** (Completed 2026-02-01)
   - Per-subject Erlang gamma kernel fitting
   - Parameters: t_max (300-1200ms), n (3-25)
   - New metrics: `kernel_t_max_ms`, `kernel_n`, `regression_r_squared_convolved`
   - Files: `src/analysis/pupil_luminance_kernel.py`, modified `whole_session_analysis.py`

2. **Cohort Analysis Pipeline** (Completed)
   - Cross-subject aggregation and statistics
   - Duration regression analysis with kernel metrics
   - Visualizations including kernel fit plots

### Current Performance

| Metric | Value |
|--------|-------|
| Instantaneous R² (median) | 0.006 |
| Convolved R² (median) | 0.007 |
| Subjects with R² > 0.05 | 2 of 13 |
| R² improvement from kernel | +0.013 (mean) |

### Root Cause Analysis

The fundamental problem: **Mean frame luminance is the wrong predictor.**

The pupil responds to light at the **point of gaze** (foveal illumination), not the average brightness of the entire surgical field. A surgeon fixating on dark tissue versus bright metallic instruments will have dramatically different retinal illumination even if mean frame luminance is identical.

---

## Recommended Implementation Path

### Phase 1: Gaze-Contingent Luminance (IMMEDIATE PRIORITY)

**Objective:** Extract luminance at the point of gaze rather than mean frame luminance.

**Expected Improvement:** R² from 0.007 → 0.05-0.20

#### Implementation Details

**New File:** `src/processing/gaze_luminance.py`

```python
def compute_gaze_luminance(frame, gaze_x, gaze_y, method='gaussian', sigma_pixels=50):
    """
    Compute luminance at gaze point with spatial weighting.

    Parameters
    ----------
    frame : np.ndarray
        Video frame (BGR or grayscale)
    gaze_x, gaze_y : float
        Gaze position in pixels
    method : str
        'gaussian' - Gaussian-weighted luminance (more accurate)
        'roi' - Simple circular ROI mean (faster)
    sigma_pixels : float
        Gaussian sigma or ROI radius (~50 pixels ≈ 2° visual angle)

    Returns
    -------
    float
        Weighted luminance at gaze point
    """
```

**Modify:** `src/processing/create_final_csv_refactored.py`
- Add `gaze_luminance` column to final CSV
- Requires loading video frames (already done for `frame_luminance`)

**Modify:** `src/analysis/whole_session_analysis.py`
- Use `gaze_luminance` instead of `frame_luminance` when available
- Keep `frame_luminance` as fallback

#### Validation Plan
1. Process one subject with gaze-contingent luminance
2. Compare R² values: frame_luminance vs gaze_luminance
3. Visualize time series overlay
4. If R² improves significantly, batch process all subjects

#### Estimated Effort: 1-2 days

---

### Phase 2: Pre-trained Feature Extraction (SHORT-TERM)

**Objective:** Use features from pre-trained video transformers as predictors of pupil response.

**Expected Improvement:** R² from 0.05 → 0.10-0.30

#### Approach

1. Use pre-trained backbone (VideoMAE, TimeSformer, or DINO-ViT)
2. Extract features for each frame
3. Pool features at gaze location (gaze-weighted spatial pooling)
4. Train lightweight temporal model: features → pupil

#### Implementation Details

**New File:** `src/analysis/video_feature_extraction.py`

```python
class VideoFeatureExtractor:
    """Extract features from pre-trained video transformers."""

    def __init__(self, model_name='facebook/videomae-base'):
        self.model = load_pretrained_model(model_name)
        self.model.eval()  # Freeze weights

    def extract_frame_features(self, frames, gaze_positions):
        """
        Extract gaze-pooled features from video frames.

        Returns feature vectors aligned with pupil time series.
        """
```

**New File:** `src/analysis/feature_based_pupil_model.py`

```python
class FeatureBasedPupilModel:
    """Predict pupil from video features."""

    def __init__(self):
        self.temporal_model = TemporalConvNet()  # or LSTM/Transformer

    def fit(self, features, pupil, timestamps):
        """Train temporal model on feature → pupil mapping."""

    def predict(self, features):
        """Predict pupil diameter from features."""
```

#### Dependencies
- `transformers` (Hugging Face)
- `torch` or `tensorflow`
- GPU recommended but not required for inference

#### Estimated Effort: 2-3 weeks

---

### Phase 3: Custom Gaze-Guided Transformer (LONG-TERM)

**Objective:** Train a custom transformer that learns the video → pupil mapping end-to-end.

**Expected Improvement:** R² potentially 0.30-0.60

#### Architecture: Gaze-Guided Video Transformer

```
INPUT:
  - Video frames: (T × H × W × 3)
  - Gaze positions: (T × 2)

PROCESSING:
  1. Gaze-guided ROI extraction (foveal patches)
  2. Spatial transformer (within-frame attention)
  3. Temporal transformer (across-frame attention, learns PLR dynamics)
  4. Cross-attention: scene context ↔ foveal content

OUTPUT:
  - Pupil diameter: (T × 1)
```

#### Data Requirements

| Current | Minimum for Training | Recommended |
|---------|---------------------|-------------|
| 14 subjects | 30-50 subjects | 100+ subjects |
| 2.7 hours | 10-20 hours | 50+ hours |
| ~900K samples | ~3M samples | ~15M samples |

#### Implementation Considerations
- Consider collaboration to expand dataset
- Use heavy data augmentation (temporal shifts, brightness jitter)
- Start with frozen pre-trained backbone, fine-tune gradually
- Cross-subject validation critical to assess generalization

#### Estimated Effort: 2-3 months

---

### Phase 4: Alternative Approaches (IF NEEDED)

If gaze-contingent luminance doesn't improve R² sufficiently:

1. **Luminance Change (ΔL):** Pupil responds to changes, not absolute levels
2. **Last Fixation Luminance:** Account for saccadic suppression
3. **Multi-scale Model:** Separate foveal and peripheral channels
4. **Saliency-Based:** Use pre-trained saliency models as intermediate representation

---

## Files to Create/Modify

### Phase 1 (Gaze-Contingent Luminance)

| File | Action | Priority |
|------|--------|----------|
| `src/processing/gaze_luminance.py` | CREATE | High |
| `src/processing/create_final_csv_refactored.py` | MODIFY - add gaze_luminance | High |
| `src/analysis/whole_session_analysis.py` | MODIFY - use gaze_luminance | High |
| `src/analysis/pupil_luminance_kernel.py` | MODIFY - support gaze_luminance | Medium |

### Phase 2 (Pre-trained Features)

| File | Action | Priority |
|------|--------|----------|
| `src/analysis/video_feature_extraction.py` | CREATE | Medium |
| `src/analysis/feature_based_pupil_model.py` | CREATE | Medium |
| `requirements.txt` | MODIFY - add transformers, torch | Medium |

### Phase 3 (Custom Transformer)

| File | Action | Priority |
|------|--------|----------|
| `src/models/gaze_guided_transformer.py` | CREATE | Low |
| `src/training/train_pupil_model.py` | CREATE | Low |
| `src/training/data_loader.py` | CREATE | Low |

---

## Success Metrics

| Phase | Target R² | Validation |
|-------|-----------|------------|
| 1 (Gaze luminance) | > 0.05 median | LOO cross-validation |
| 2 (Pre-trained features) | > 0.15 median | Subject holdout |
| 3 (Custom transformer) | > 0.30 median | External dataset |

---

## Research References

### Foundational
1. Hoeks & Levelt (1993) - Erlang gamma PLR model
2. Mathôt (2018) - Pupillometry review

### Transformer-Based Approaches
3. [ViV1T](https://www.biorxiv.org/content/10.1101/2025.09.16.676524v1.full) - Movie-trained transformer with pupil data
4. [Foundation Model for Neural Activity](https://www.nature.com/articles/s41586-025-08829-y) - Video → neural response
5. [Visual Saliency Transformer](https://arxiv.org/abs/2104.12099) - Pure transformer for saliency

### Remote Physiological Sensing
6. [rPPG Review](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1420100/full) - Deep learning for video-based physiological signals
7. [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) - Open-source implementations

### Pupil and Saliency
8. [Pupil and Contrast Saliency](https://www.jneurosci.org/content/34/2/408) - Saliency modulates pupil response

---

## Appendix: Key Findings from Current Analysis

### Duration Regression with Kernel Metrics

The convolved R² (`regression_r_squared_convolved`) was selected by Lasso as a predictor of recording duration, but this correlation was driven by 2 outlier subjects. Excluding them inverts the correlation from r=+0.53 to r=-0.57.

### Kernel Parameter Distribution

| Parameter | Mean ± SD | Range |
|-----------|-----------|-------|
| t_max (ms) | 685 ± 271 | 400-1200 |
| n (shape) | 7.9 ± 3.5 | 3.0-12.0 |
| Fitting success | 79% | - |

### Subjects with Highest R²

Only 2 subjects had R² > 0.10:
- 20251029T125515Z: R² = 0.19
- 20231027T171918Z: R² = 0.13

The remaining 11 subjects had R² < 0.05, indicating that temporal convolution alone is insufficient without addressing the fundamental luminance measurement issue.

---

## Next Action Items

1. [ ] **Implement gaze-contingent luminance extraction** (Phase 1)
2. [ ] **Test on single subject** to validate improvement
3. [ ] **Batch process all subjects** if validation successful
4. [ ] **Update cohort analysis** with new metrics
5. [ ] **Re-run duration regression** with corrected luminance
6. [ ] **Document findings** and decide on Phase 2 timeline
