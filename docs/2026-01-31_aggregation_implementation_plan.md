# Cross-Subject Aggregation Implementation Plan

**Date:** 2026-01-31
**Status:** Ready for Implementation
**Goal:** Finalize the pipeline by adding cross-subject aggregation and comparison capabilities

---

## Problem Statement

Individual subject analyses are complete (14 subjects processed), but:
- No aggregation across subjects exists
- No cohort-level statistics are computed
- No cross-subject comparison visualizations
- No unified export for statistical analysis
- Batch summary only lists individual results without pooled metrics

---

## Implementation Plan

### Phase 1: Create CohortAnalyzer Module

**New file:** `src/analysis/cohort_analyzer.py`

```python
class CohortAnalyzer:
    """Aggregates and compares metrics across all subjects."""

    def __init__(self, logs_dir: str, figures_dir: str):
        self.logs_dir = logs_dir
        self.figures_dir = figures_dir
        self.subjects_data = {}  # subject_id -> metrics dict

    def load_all_subjects(self) -> pd.DataFrame:
        """Load all *_whole_session_analysis.json files into unified DataFrame."""

    def compute_cohort_statistics(self) -> Dict:
        """Calculate mean, std, min, max, median for each metric across subjects."""

    def identify_outliers(self, method='iqr') -> Dict:
        """Flag subjects with metrics outside 1.5*IQR."""

    def export_to_csv(self, output_path: str) -> None:
        """Export subjects × metrics matrix to CSV for external analysis."""

    def generate_cohort_report(self) -> str:
        """Generate human-readable summary report."""
```

**Key metrics to aggregate (from whole_session_analysis.json):**

| Goal | Metric | Description |
|------|--------|-------------|
| A | fixation_rate_hz | Fixations per second |
| A | mean_fixation_duration_ms | Average fixation duration |
| A | saccade_count | Valid saccades detected |
| A | adaptive_threshold_deg_s | MAD-computed threshold |
| A | saccade_amplitude_mean_deg | Mean saccade amplitude |
| A | main_sequence_r_squared | Main sequence fit quality |
| B | std_residual | Pupil variability (cognitive load proxy) |
| B | regression_r_squared | Luminance-pupil correlation |
| B | raw_pupil_mean | Average pupil size |
| C | total_rotation_deg | Total head movement |
| C | mean_rotation_rate | Average rotation speed |
| C | rotation_rate_per_second | Normalized head motion |

---

### Phase 2: Create Cohort Visualizations

**New file:** `src/analysis/cohort_visualizations.py`

```python
class CohortVisualizer:
    """Creates cross-subject comparison visualizations."""

    def create_metric_comparison_boxplots(self, df: pd.DataFrame, output_path: str):
        """Box plots comparing each metric across all subjects."""

    def create_subject_ranking_chart(self, df: pd.DataFrame, metric: str, output_path: str):
        """Horizontal bar chart ranking subjects by a specific metric."""

    def create_correlation_heatmap(self, df: pd.DataFrame, output_path: str):
        """Heatmap showing correlations between all metrics."""

    def create_cohort_dashboard(self, df: pd.DataFrame, output_path: str):
        """Multi-panel summary dashboard with key metrics."""

    def create_main_sequence_overlay(self, logs_dir: str, output_path: str):
        """Overlay all subjects' main sequence data on single plot."""
```

**Visualizations to generate:**

1. **Metric Distribution Box Plots** (`cohort_metric_distributions.png`)
   - 3×4 grid of box plots, one per metric
   - Shows spread and outliers across subjects

2. **Subject Ranking Charts** (`cohort_rankings_*.png`)
   - Horizontal bar charts for key metrics
   - Identifies best/worst performers

3. **Correlation Heatmap** (`cohort_correlation_matrix.png`)
   - Shows relationships between metrics
   - Identifies redundant vs independent measures

4. **Cohort Dashboard** (`cohort_summary_dashboard.png`)
   - Combined visualization with:
     - Sample size and duration summary
     - Key metric distributions
     - Data quality overview

5. **Main Sequence Overlay** (`cohort_main_sequence_overlay.png`)
   - All subjects' saccade data on one plot
   - Different colors per subject
   - Shows population-level main sequence relationship

---

### Phase 3: Integrate with Batch Processing

**Modify:** `src/batch_process_with_heatmaps.py`

Add new step after individual processing:

```python
def run_cohort_analysis(self):
    """Step 6: Cross-subject aggregation and comparison."""
    from src.analysis.cohort_analyzer import CohortAnalyzer
    from src.analysis.cohort_visualizations import CohortVisualizer

    analyzer = CohortAnalyzer(self.logs_dir, self.figures_dir)
    df = analyzer.load_all_subjects()

    # Generate outputs
    analyzer.export_to_csv(self.logs_dir / 'cohort_metrics_matrix.csv')
    stats = analyzer.compute_cohort_statistics()

    visualizer = CohortVisualizer()
    visualizer.create_all_visualizations(df, self.figures_dir)

    return analyzer.generate_cohort_report()
```

---

### Phase 4: Output Structure

**New files to generate:**

```
reports/
├── logs/
│   ├── cohort_metrics_matrix.csv      # Subjects × metrics spreadsheet
│   ├── cohort_statistics.json         # Aggregated stats (mean, std, etc.)
│   └── cohort_analysis_report.txt     # Human-readable summary
├── figures/
│   ├── cohort_metric_distributions.png
│   ├── cohort_correlation_matrix.png
│   ├── cohort_summary_dashboard.png
│   ├── cohort_main_sequence_overlay.png
│   ├── cohort_ranking_fixation_rate.png
│   ├── cohort_ranking_cognitive_load.png
│   └── cohort_ranking_motor_stability.png
```

---

## Implementation Checklist

- [ ] Create `src/analysis/cohort_analyzer.py`
  - [ ] `load_all_subjects()` - Load all JSON files
  - [ ] `compute_cohort_statistics()` - Mean, std, min, max, median
  - [ ] `identify_outliers()` - IQR-based outlier detection
  - [ ] `export_to_csv()` - Generate metrics matrix
  - [ ] `generate_cohort_report()` - Text summary

- [ ] Create `src/analysis/cohort_visualizations.py`
  - [ ] `create_metric_comparison_boxplots()` - Distribution plots
  - [ ] `create_correlation_heatmap()` - Metric correlations
  - [ ] `create_cohort_dashboard()` - Summary dashboard
  - [ ] `create_main_sequence_overlay()` - Combined saccade plot
  - [ ] `create_subject_ranking_chart()` - Bar chart rankings

- [ ] Integrate with batch processing
  - [ ] Add `run_cohort_analysis()` method
  - [ ] Update batch summary to include cohort stats

- [ ] Test and validate
  - [ ] Run on all 14 subjects
  - [ ] Verify CSV export opens in Excel/SPSS
  - [ ] Check all visualizations render correctly

---

## CSV Export Schema

**File:** `cohort_metrics_matrix.csv`

| subject_id | recording_duration_s | fixation_rate_hz | mean_fixation_duration_ms | saccade_count | adaptive_threshold_deg_s | saccade_amplitude_mean_deg | main_sequence_r_squared | pupil_residual_std | pupil_luminance_r2 | raw_pupil_mean | total_rotation_deg | mean_rotation_rate | rotation_rate_per_second |
|------------|---------------------|------------------|--------------------------|---------------|-------------------------|---------------------------|------------------------|-------------------|-------------------|----------------|-------------------|-------------------|-------------------------|
| 20231027T170020Z | 740.2 | 8.11 | 91.0 | 1761 | 38.6 | 3.7 | 0.796 | 0.194 | 0.006 | 3.49 | 9795 | 13.23 | 13.23 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## Expected Outcomes

1. **Unified metrics matrix** - Single CSV with all subjects and metrics for statistical analysis
2. **Cohort statistics** - Mean ± SD for each metric across the cohort
3. **Comparison visualizations** - Box plots, rankings, correlations
4. **Quality assessment** - Outlier identification
5. **Main sequence population fit** - Combined R² across all subjects

---

## Verification Steps

After implementation:
1. Run `python -m src.batch_process_with_heatmaps` with cohort analysis enabled
2. Verify `cohort_metrics_matrix.csv` contains all 14 subjects × 13+ metrics
3. Open CSV in Excel to confirm compatibility
4. Check all `cohort_*.png` files generated in `reports/figures/`
5. Review `cohort_analysis_report.txt` for cohort summary
