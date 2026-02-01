"""
Cohort Analysis Module for Cross-Subject Aggregation

Aggregates metrics from individual subject analyses into unified
cohort-level statistics and exports for external statistical analysis.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..logging_config import get_logger

logger = get_logger(__name__)


class CohortAnalyzer:
    """
    Aggregates and compares metrics across all subjects.

    Loads individual whole_session_analysis.json files and creates:
    - Unified DataFrame with all subjects × all metrics
    - Cohort-level statistics (mean, std, min, max, median)
    - Outlier detection
    - CSV export for external analysis
    """

    # Metrics to extract from each goal
    METRICS_SCHEMA = {
        'goal_a_oculometric_efficiency': [
            ('fixation_rate_hz', 'Fixation Rate (Hz)'),
            ('fixation_count', 'Fixation Count'),
            ('fixation_proportion', 'Fixation Proportion'),
            ('mean_fixation_duration_ms', 'Mean Fixation Duration (ms)'),
            ('saccade_count', 'Saccade Count'),
            ('saccade_proportion', 'Saccade Proportion'),
            ('total_duration_s', 'Recording Duration (s)'),
            ('adaptive_threshold_deg_s', 'Adaptive Threshold (deg/s)'),
            ('saccade_amplitude_mean_deg', 'Saccade Amplitude Mean (deg)'),
            ('saccade_amplitude_max_deg', 'Saccade Amplitude Max (deg)'),
            ('saccade_peak_velocity_mean_deg_s', 'Peak Velocity Mean (deg/s)'),
            ('saccade_peak_velocity_max_deg_s', 'Peak Velocity Max (deg/s)'),
            ('main_sequence_r_squared', 'Main Sequence R²'),
        ],
        'goal_b_cognitive_load': [
            ('mean_residual', 'Pupil Residual Mean (mm)'),
            ('std_residual', 'Pupil Residual Std (mm)'),
            ('regression_r_squared', 'Luminance-Pupil R² (instantaneous)'),
            ('regression_r_squared_convolved', 'Luminance-Pupil R² (convolved)'),
            ('regression_slope', 'Luminance-Pupil Slope'),
            ('kernel_t_max_ms', 'PLR Kernel t_max (ms)'),
            ('kernel_n', 'PLR Kernel n (shape)'),
            ('kernel_is_fitted', 'PLR Kernel Fitted'),
            ('r_squared_improvement', 'R² Improvement (convolved - inst)'),
            ('r_squared_improvement_pct', 'R² Improvement (%)'),
            ('raw_pupil_mean', 'Raw Pupil Mean (mm)'),
            ('raw_pupil_std', 'Raw Pupil Std (mm)'),
            ('luminance_mean', 'Frame Luminance Mean'),
            ('n_samples', 'Valid Pupil Samples'),
        ],
        'goal_c_motor_stability': [
            ('total_rotation_deg', 'Total Head Rotation (deg)'),
            ('mean_rotation_rate', 'Mean Rotation Rate (deg/s)'),
            ('rotation_rate_std', 'Rotation Rate Std (deg/s)'),
            ('rotation_rate_max', 'Max Rotation Rate (deg/s)'),
            ('gyro_x_mean', 'Gyro X Mean (deg/s)'),
            ('gyro_y_mean', 'Gyro Y Mean (deg/s)'),
            ('gyro_z_mean', 'Gyro Z Mean (deg/s)'),
            ('rotation_rate_per_second', 'Normalized Rotation Rate'),
        ],
    }

    def __init__(self, logs_dir: str, figures_dir: str = None):
        """
        Initialize the cohort analyzer.

        Args:
            logs_dir: Directory containing *_whole_session_analysis.json files
            figures_dir: Directory for output figures (optional)
        """
        self.logs_dir = Path(logs_dir)
        self.figures_dir = Path(figures_dir) if figures_dir else None
        self.subjects_data = {}
        self.df = None
        self.statistics = None

    def load_all_subjects(self) -> pd.DataFrame:
        """
        Load all whole_session_analysis.json files into unified DataFrame.

        Returns:
            DataFrame with subjects as rows and metrics as columns
        """
        logger.info(f"Loading subject data from: {self.logs_dir}")

        # Find all analysis JSON files
        json_files = list(self.logs_dir.glob("*_whole_session_analysis.json"))
        logger.info(f"Found {len(json_files)} analysis files")

        if not json_files:
            logger.warning("No whole_session_analysis.json files found")
            return pd.DataFrame()

        records = []

        for json_path in sorted(json_files):
            subject_id = json_path.stem.replace("_whole_session_analysis", "")

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Extract metrics from each goal
                record = {'subject_id': subject_id}

                for goal_key, metrics in self.METRICS_SCHEMA.items():
                    goal_data = data.get(goal_key, {})
                    if goal_data is None:
                        goal_data = {}

                    for metric_key, metric_label in metrics:
                        value = goal_data.get(metric_key)
                        record[metric_key] = value

                records.append(record)
                self.subjects_data[subject_id] = data
                logger.debug(f"  Loaded: {subject_id}")

            except Exception as e:
                logger.warning(f"  Failed to load {json_path.name}: {e}")

        # Create DataFrame
        self.df = pd.DataFrame(records)
        self.df.set_index('subject_id', inplace=True)

        logger.info(f"Loaded {len(self.df)} subjects with {len(self.df.columns)} metrics")

        return self.df

    def compute_cohort_statistics(self) -> Dict:
        """
        Calculate descriptive statistics for each metric across all subjects.

        Returns:
            Dictionary with statistics for each metric
        """
        if self.df is None or self.df.empty:
            logger.warning("No data loaded. Call load_all_subjects() first.")
            return {}

        logger.info("Computing cohort statistics...")

        stats = {
            'n_subjects': len(self.df),
            'metrics': {}
        }

        for col in self.df.columns:
            values = self.df[col].dropna()

            if len(values) == 0:
                continue

            # Get human-readable label
            label = col
            for goal_metrics in self.METRICS_SCHEMA.values():
                for key, lbl in goal_metrics:
                    if key == col:
                        label = lbl
                        break

            # Handle boolean columns differently
            if values.dtype == 'bool' or col.endswith('_fitted'):
                stats['metrics'][col] = {
                    'label': label,
                    'n_valid': len(values),
                    'true_count': int(values.sum()),
                    'false_count': int((~values).sum()),
                    'true_proportion': float(values.mean()),
                }
            else:
                try:
                    stats['metrics'][col] = {
                        'label': label,
                        'n_valid': len(values),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                        'q1': float(values.quantile(0.25)),
                        'q3': float(values.quantile(0.75)),
                    }
                except (TypeError, ValueError) as e:
                    logger.debug(f"Skipping non-numeric column {col}: {e}")

        self.statistics = stats
        logger.info(f"Computed statistics for {len(stats['metrics'])} metrics")

        return stats

    def identify_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict:
        """
        Identify subjects with outlier values for each metric.

        Args:
            method: 'iqr' (interquartile range) or 'zscore'
            threshold: IQR multiplier (default 1.5) or z-score threshold

        Returns:
            Dictionary mapping metric -> list of outlier subjects
        """
        if self.df is None or self.df.empty:
            return {}

        logger.info(f"Identifying outliers using {method} method...")

        outliers = {}

        for col in self.df.columns:
            values = self.df[col].dropna()

            if len(values) < 4:
                continue

            # Skip boolean columns
            if values.dtype == 'bool' or col.endswith('_fitted'):
                continue

            if method == 'iqr':
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr

                outlier_mask = (self.df[col] < lower) | (self.df[col] > upper)

            elif method == 'zscore':
                mean = values.mean()
                std = values.std()
                if std > 0:
                    z_scores = (self.df[col] - mean) / std
                    outlier_mask = z_scores.abs() > threshold
                else:
                    outlier_mask = pd.Series(False, index=self.df.index)
            else:
                raise ValueError(f"Unknown method: {method}")

            outlier_subjects = self.df.index[outlier_mask].tolist()

            if outlier_subjects:
                outliers[col] = outlier_subjects

        n_metrics_with_outliers = len(outliers)
        n_total_outliers = sum(len(v) for v in outliers.values())
        logger.info(f"Found outliers in {n_metrics_with_outliers} metrics "
                   f"({n_total_outliers} total outlier instances)")

        return outliers

    def export_to_csv(self, output_path: str) -> None:
        """
        Export the subjects × metrics matrix to CSV.

        Args:
            output_path: Path for output CSV file
        """
        if self.df is None or self.df.empty:
            logger.warning("No data to export")
            return

        output_path = Path(output_path)

        # Reset index to include subject_id as column
        export_df = self.df.reset_index()

        # Reorder columns for readability
        priority_cols = [
            'subject_id',
            'total_duration_s',
            'fixation_rate_hz',
            'mean_fixation_duration_ms',
            'saccade_count',
            'adaptive_threshold_deg_s',
            'saccade_amplitude_mean_deg',
            'main_sequence_r_squared',
            'std_residual',
            'raw_pupil_mean',
            'total_rotation_deg',
            'mean_rotation_rate',
        ]

        # Build column order: priority cols first, then remaining
        ordered_cols = [c for c in priority_cols if c in export_df.columns]
        remaining_cols = [c for c in export_df.columns if c not in ordered_cols]
        export_df = export_df[ordered_cols + remaining_cols]

        export_df.to_csv(output_path, index=False)
        logger.info(f"Exported cohort matrix to: {output_path}")
        logger.info(f"  Shape: {len(export_df)} subjects × {len(export_df.columns)} columns")

    def export_statistics_json(self, output_path: str) -> None:
        """
        Export cohort statistics to JSON file.

        Args:
            output_path: Path for output JSON file
        """
        if self.statistics is None:
            self.compute_cohort_statistics()

        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)

        logger.info(f"Exported statistics to: {output_path}")

    def generate_cohort_report(self) -> str:
        """
        Generate a human-readable summary report.

        Returns:
            Formatted string report
        """
        if self.df is None or self.df.empty:
            return "No data loaded."

        if self.statistics is None:
            self.compute_cohort_statistics()

        lines = []
        lines.append("=" * 70)
        lines.append("COHORT ANALYSIS REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        # Overview
        lines.append(f"\nSUBJECTS: {len(self.df)}")
        lines.append(f"METRICS: {len(self.df.columns)}")

        # Recording duration summary
        if 'total_duration_s' in self.df.columns:
            durations = self.df['total_duration_s'].dropna()
            total_hours = durations.sum() / 3600
            lines.append(f"TOTAL RECORDING TIME: {total_hours:.2f} hours")
            lines.append(f"MEAN DURATION: {durations.mean():.1f} s ({durations.mean()/60:.1f} min)")

        # Goal A: Oculometric Efficiency
        lines.append("\n" + "-" * 70)
        lines.append("GOAL A: OCULOMETRIC EFFICIENCY")
        lines.append("-" * 70)

        key_metrics_a = [
            ('fixation_rate_hz', 'Fixation Rate', 'Hz'),
            ('mean_fixation_duration_ms', 'Mean Fixation Duration', 'ms'),
            ('saccade_count', 'Valid Saccade Count', ''),
            ('adaptive_threshold_deg_s', 'Adaptive Threshold', 'deg/s'),
            ('saccade_amplitude_mean_deg', 'Saccade Amplitude', 'deg'),
            ('main_sequence_r_squared', 'Main Sequence R²', ''),
        ]

        for metric, label, unit in key_metrics_a:
            if metric in self.statistics['metrics']:
                s = self.statistics['metrics'][metric]
                unit_str = f" {unit}" if unit else ""
                lines.append(f"  {label}: {s['mean']:.2f} +/- {s['std']:.2f}{unit_str} "
                           f"(range: {s['min']:.2f} - {s['max']:.2f})")

        # Goal B: Cognitive Load
        lines.append("\n" + "-" * 70)
        lines.append("GOAL B: COGNITIVE LOAD (Pupil Analysis)")
        lines.append("-" * 70)

        key_metrics_b = [
            ('std_residual', 'Pupil Residual Std', 'mm'),
            ('raw_pupil_mean', 'Raw Pupil Mean', 'mm'),
            ('regression_r_squared', 'R² Instantaneous', ''),
            ('regression_r_squared_convolved', 'R² Convolved', ''),
            ('r_squared_improvement', 'R² Improvement', ''),
        ]

        for metric, label, unit in key_metrics_b:
            if metric in self.statistics['metrics']:
                s = self.statistics['metrics'][metric]
                unit_str = f" {unit}" if unit else ""
                lines.append(f"  {label}: {s['mean']:.4f} +/- {s['std']:.4f}{unit_str} "
                           f"(range: {s['min']:.4f} - {s['max']:.4f})")

        # PLR Kernel Parameters
        lines.append("\n  PLR Temporal Kernel (per-subject fitted):")
        kernel_metrics = [
            ('kernel_t_max_ms', 't_max (time to peak)', 'ms'),
            ('kernel_n', 'n (shape parameter)', ''),
        ]

        for metric, label, unit in kernel_metrics:
            if metric in self.statistics['metrics']:
                s = self.statistics['metrics'][metric]
                unit_str = f" {unit}" if unit else ""
                lines.append(f"    {label}: {s['mean']:.1f} +/- {s['std']:.1f}{unit_str} "
                           f"(range: {s['min']:.1f} - {s['max']:.1f})")

        # Goal C: Motor Stability
        lines.append("\n" + "-" * 70)
        lines.append("GOAL C: MOTOR STABILITY (Head Movement)")
        lines.append("-" * 70)

        key_metrics_c = [
            ('total_rotation_deg', 'Total Head Rotation', 'deg'),
            ('mean_rotation_rate', 'Mean Rotation Rate', 'deg/s'),
            ('rotation_rate_per_second', 'Normalized Rate', 'deg/s'),
        ]

        for metric, label, unit in key_metrics_c:
            if metric in self.statistics['metrics']:
                s = self.statistics['metrics'][metric]
                unit_str = f" {unit}" if unit else ""
                lines.append(f"  {label}: {s['mean']:.1f} +/- {s['std']:.1f}{unit_str} "
                           f"(range: {s['min']:.1f} - {s['max']:.1f})")

        # Outliers
        outliers = self.identify_outliers()
        if outliers:
            lines.append("\n" + "-" * 70)
            lines.append("OUTLIER SUBJECTS (>1.5 IQR)")
            lines.append("-" * 70)
            for metric, subjects in outliers.items():
                lines.append(f"  {metric}: {', '.join(subjects)}")

        # Subject list
        lines.append("\n" + "-" * 70)
        lines.append("SUBJECTS INCLUDED")
        lines.append("-" * 70)
        for i, subject in enumerate(sorted(self.df.index), 1):
            dur = self.df.loc[subject, 'total_duration_s'] if 'total_duration_s' in self.df.columns else 0
            dur_str = f" ({dur:.0f}s)" if dur else ""
            lines.append(f"  {i:2d}. {subject}{dur_str}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def save_report(self, output_path: str) -> None:
        """
        Save the cohort report to a text file.

        Args:
            output_path: Path for output text file
        """
        report = self.generate_cohort_report()

        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Saved cohort report to: {output_path}")

    def get_metric_values(self, metric: str) -> pd.Series:
        """
        Get values for a specific metric across all subjects.

        Args:
            metric: Metric column name

        Returns:
            Series with subject_id index and metric values
        """
        if self.df is None or metric not in self.df.columns:
            return pd.Series()

        return self.df[metric].dropna()

    def get_subject_profile(self, subject_id: str) -> Dict:
        """
        Get all metrics for a specific subject.

        Args:
            subject_id: Subject identifier

        Returns:
            Dictionary of metric -> value
        """
        if self.df is None or subject_id not in self.df.index:
            return {}

        return self.df.loc[subject_id].to_dict()


def run_cohort_analysis(
    logs_dir: str,
    output_dir: str = None
) -> Tuple[pd.DataFrame, Dict, str]:
    """
    Convenience function to run complete cohort analysis.

    Args:
        logs_dir: Directory containing analysis JSON files
        output_dir: Directory for outputs (defaults to logs_dir)

    Returns:
        Tuple of (DataFrame, statistics dict, report string)
    """
    output_dir = Path(output_dir or logs_dir)

    analyzer = CohortAnalyzer(logs_dir)
    df = analyzer.load_all_subjects()

    if df.empty:
        return df, {}, "No subjects found"

    # Compute statistics
    stats = analyzer.compute_cohort_statistics()

    # Export outputs
    analyzer.export_to_csv(output_dir / "cohort_metrics_matrix.csv")
    analyzer.export_statistics_json(output_dir / "cohort_statistics.json")
    analyzer.save_report(output_dir / "cohort_analysis_report.txt")

    report = analyzer.generate_cohort_report()

    return df, stats, report
