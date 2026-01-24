"""
Whole-Session Analysis Module for Surgical Skill Assessment

Implements the "Whole-Session" Global Analysis goals from the revised roadmap:
- Goal A: Oculometric Efficiency (Global Fixation Rate)
- Goal B: Cognitive Load (Luminance-Adjusted Pupil Residuals)
- Goal C: Motor Stability (Integrated Gyroscopic Motion)
- Goal D: Visual Strategy (Tool vs Tissue Classification)

This module operates on the enhanced final_gaze_data.csv which contains:
- angular_velocity_deg_s: Physics-based angular velocity
- pupil_diameter_avg: Average pupil diameter
- frame_luminance: Frame brightness
- head_gyro_x/y/z: IMU gyroscope data
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import cv2
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..logging_config import get_logger

logger = get_logger(__name__)


class WholeSessionAnalyzer:
    """
    Analyzes entire recording sessions for surgical skill metrics.

    Since task-specific timestamps are not available, this analyzer
    treats the entire recording as a single performance block.
    """

    # I-VT velocity thresholds (degrees/second)
    FIXATION_THRESHOLD = 30.0    # Below this = fixation
    SACCADE_THRESHOLD = 300.0    # Above this = saccade

    def __init__(self, csv_path: str = None, video_path: str = None):
        """
        Initialize the analyzer.

        Args:
            csv_path: Path to enhanced final_gaze_data.csv
            video_path: Path to scenevideo.mp4 (for Goal D analysis)
        """
        self.csv_path = csv_path
        self.video_path = video_path
        self.data = None
        self.metrics = {}

    def load_data(self, csv_path: str = None) -> pd.DataFrame:
        """
        Load the enhanced gaze data CSV.

        Args:
            csv_path: Path to CSV file (overrides constructor path)

        Returns:
            DataFrame with gaze data
        """
        path = csv_path or self.csv_path
        if path is None:
            raise ValueError("No CSV path provided")

        logger.info(f"Loading gaze data from: {path}")
        self.data = pd.read_csv(path)

        logger.info(f"Loaded {len(self.data)} records")
        logger.info(f"Columns: {list(self.data.columns)}")

        # Validate required columns
        required = ['gaze_timestamp', 'angular_velocity_deg_s']
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")

        return self.data

    # ==================== GOAL A: Oculometric Efficiency ====================

    def calculate_global_fixation_rate(self) -> Dict[str, float]:
        """
        Goal A: Calculate global fixation rate using angular velocity I-VT.

        Hypothesis: Does the surgeon fixate frequently (novice searching)
        or stably (expert planning)?

        Returns:
            Dictionary with fixation metrics:
            - fixation_rate_hz: Fixations per second
            - fixation_count: Total number of fixation events
            - fixation_proportion: Proportion of time in fixation state
            - mean_fixation_duration_ms: Average fixation duration
            - saccade_count: Number of saccade events
            - saccade_proportion: Proportion of time in saccade state
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Calculating global fixation rate (Goal A)...")

        # Get valid angular velocity data
        mask = self.data['angular_velocity_deg_s'].notna()
        valid_data = self.data[mask].copy()

        if len(valid_data) < 10:
            logger.warning("Insufficient valid angular velocity data")
            return self._empty_fixation_metrics()

        # Classify each sample using I-VT thresholds
        velocities = valid_data['angular_velocity_deg_s'].values
        timestamps = valid_data['gaze_timestamp'].values

        # Classify states
        states = np.empty(len(velocities), dtype=object)
        states[velocities < self.FIXATION_THRESHOLD] = 'FIXATION'
        states[velocities >= self.SACCADE_THRESHOLD] = 'SACCADE'
        states[(velocities >= self.FIXATION_THRESHOLD) &
               (velocities < self.SACCADE_THRESHOLD)] = 'OTHER'

        # Count fixation events (transitions into fixation state)
        fixation_starts = (states == 'FIXATION') & (np.roll(states, 1) != 'FIXATION')
        fixation_starts[0] = states[0] == 'FIXATION'  # Handle first sample
        fixation_count = fixation_starts.sum()

        # Count saccade events
        saccade_starts = (states == 'SACCADE') & (np.roll(states, 1) != 'SACCADE')
        saccade_starts[0] = states[0] == 'SACCADE'
        saccade_count = saccade_starts.sum()

        # Calculate total duration
        total_duration = timestamps[-1] - timestamps[0]

        # Calculate fixation rate (Hz)
        fixation_rate_hz = fixation_count / total_duration if total_duration > 0 else 0

        # Calculate proportions
        fixation_samples = (states == 'FIXATION').sum()
        saccade_samples = (states == 'SACCADE').sum()
        total_samples = len(states)

        fixation_proportion = fixation_samples / total_samples
        saccade_proportion = saccade_samples / total_samples

        # Calculate mean fixation duration
        fixation_durations = self._calculate_event_durations(
            states, timestamps, 'FIXATION'
        )
        mean_fixation_duration_ms = (
            np.mean(fixation_durations) * 1000 if len(fixation_durations) > 0 else 0
        )

        metrics = {
            'fixation_rate_hz': fixation_rate_hz,
            'fixation_count': int(fixation_count),
            'fixation_proportion': fixation_proportion,
            'mean_fixation_duration_ms': mean_fixation_duration_ms,
            'saccade_count': int(saccade_count),
            'saccade_proportion': saccade_proportion,
            'total_duration_s': total_duration,
            'total_samples': total_samples
        }

        self.metrics['goal_a'] = metrics
        logger.info(f"  Fixation rate: {fixation_rate_hz:.2f} Hz")
        logger.info(f"  Fixation count: {fixation_count}")
        logger.info(f"  Mean fixation duration: {mean_fixation_duration_ms:.1f} ms")

        return metrics

    def _calculate_event_durations(
        self,
        states: np.ndarray,
        timestamps: np.ndarray,
        target_state: str
    ) -> List[float]:
        """Calculate durations of consecutive events of a given state."""
        durations = []
        in_event = False
        event_start = 0

        for i, state in enumerate(states):
            if state == target_state:
                if not in_event:
                    in_event = True
                    event_start = timestamps[i]
            else:
                if in_event:
                    in_event = False
                    durations.append(timestamps[i] - event_start)

        # Handle event that extends to end of recording
        if in_event:
            durations.append(timestamps[-1] - event_start)

        return durations

    def _empty_fixation_metrics(self) -> Dict[str, float]:
        """Return empty fixation metrics dictionary."""
        return {
            'fixation_rate_hz': 0.0,
            'fixation_count': 0,
            'fixation_proportion': 0.0,
            'mean_fixation_duration_ms': 0.0,
            'saccade_count': 0,
            'saccade_proportion': 0.0,
            'total_duration_s': 0.0,
            'total_samples': 0
        }

    # ==================== GOAL B: Cognitive Load ====================

    def calculate_luminance_adjusted_pupil_residuals(self) -> Dict[str, float]:
        """
        Goal B: Calculate luminance-adjusted pupil residuals as cognitive load proxy.

        Logic:
        1. Fit linear regression: Pupil Diameter ~ Luminance
        2. Calculate residuals: R = Observed - Predicted
        3. Mean R > 0 indicates high cognitive load

        Returns:
            Dictionary with pupil metrics:
            - mean_residual: Mean pupil residual (positive = high load)
            - std_residual: Std dev of residuals
            - regression_r_squared: R² of luminance regression
            - regression_slope: Slope of luminance effect
            - raw_pupil_mean: Mean raw pupil diameter (mm)
            - raw_pupil_std: Std dev of raw pupil (mm)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Calculating luminance-adjusted pupil residuals (Goal B)...")

        # Get valid data with both pupil and luminance
        mask = (
            self.data['pupil_diameter_avg'].notna() &
            self.data['frame_luminance'].notna()
        )
        valid_data = self.data[mask].copy()

        if len(valid_data) < 100:
            logger.warning("Insufficient valid pupil/luminance data for regression")
            return self._empty_pupil_metrics()

        pupil = valid_data['pupil_diameter_avg'].values
        luminance = valid_data['frame_luminance'].values

        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            luminance, pupil
        )

        # Calculate predicted values and residuals
        predicted = slope * luminance + intercept
        residuals = pupil - predicted

        # Calculate metrics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        r_squared = r_value ** 2

        metrics = {
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'regression_r_squared': r_squared,
            'regression_slope': slope,
            'regression_intercept': intercept,
            'regression_p_value': p_value,
            'raw_pupil_mean': np.mean(pupil),
            'raw_pupil_std': np.std(pupil),
            'luminance_mean': np.mean(luminance),
            'luminance_std': np.std(luminance),
            'n_samples': len(valid_data)
        }

        self.metrics['goal_b'] = metrics
        logger.info(f"  Mean residual: {mean_residual:.4f} mm")
        logger.info(f"  R² (luminance effect): {r_squared:.3f}")
        logger.info(f"  Regression slope: {slope:.6f}")

        # Interpretation
        if mean_residual > 0:
            logger.info("  Interpretation: POSITIVE residual suggests elevated cognitive load")
        else:
            logger.info("  Interpretation: NEGATIVE residual suggests lower cognitive load")

        return metrics

    def _empty_pupil_metrics(self) -> Dict[str, float]:
        """Return empty pupil metrics dictionary."""
        return {
            'mean_residual': 0.0,
            'std_residual': 0.0,
            'regression_r_squared': 0.0,
            'regression_slope': 0.0,
            'regression_intercept': 0.0,
            'regression_p_value': 1.0,
            'raw_pupil_mean': 0.0,
            'raw_pupil_std': 0.0,
            'luminance_mean': 0.0,
            'luminance_std': 0.0,
            'n_samples': 0
        }

    # ==================== GOAL C: Motor Stability ====================

    def calculate_integrated_gyro_motion(self) -> Dict[str, float]:
        """
        Goal C: Calculate integrated gyroscopic motion as motor stability metric.

        Logic: Sum of absolute rotational velocity over time.
        Lower total = steadier head = better motor control.

        Returns:
            Dictionary with IMU metrics:
            - total_rotation_deg: Total integrated rotation (degrees)
            - mean_rotation_rate: Mean rotation rate (deg/s)
            - rotation_rate_std: Std dev of rotation rate
            - x/y/z component metrics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Calculating integrated gyroscopic motion (Goal C)...")

        # Get valid IMU data
        mask = (
            self.data['head_gyro_x'].notna() &
            self.data['head_gyro_y'].notna() &
            self.data['head_gyro_z'].notna() &
            self.data['gaze_timestamp'].notna()
        )
        valid_data = self.data[mask].copy()

        if len(valid_data) < 100:
            logger.warning("Insufficient valid IMU data")
            return self._empty_imu_metrics()

        gyro_x = valid_data['head_gyro_x'].values
        gyro_y = valid_data['head_gyro_y'].values
        gyro_z = valid_data['head_gyro_z'].values
        timestamps = valid_data['gaze_timestamp'].values

        # Calculate total rotation magnitude at each sample
        rotation_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

        # Integrate rotation over time (trapezoidal integration)
        dt = np.diff(timestamps)
        dt = np.clip(dt, 0, 0.1)  # Clip to avoid outliers

        # Total integrated rotation
        integrated_rotation = np.sum(rotation_magnitude[:-1] * dt)

        # Duration
        total_duration = timestamps[-1] - timestamps[0]

        # Calculate per-axis metrics
        metrics = {
            'total_rotation_deg': integrated_rotation,
            'mean_rotation_rate': np.mean(rotation_magnitude),
            'rotation_rate_std': np.std(rotation_magnitude),
            'rotation_rate_max': np.max(rotation_magnitude),
            'gyro_x_mean': np.mean(np.abs(gyro_x)),
            'gyro_x_std': np.std(gyro_x),
            'gyro_y_mean': np.mean(np.abs(gyro_y)),
            'gyro_y_std': np.std(gyro_y),
            'gyro_z_mean': np.mean(np.abs(gyro_z)),
            'gyro_z_std': np.std(gyro_z),
            'total_duration_s': total_duration,
            'n_samples': len(valid_data),
            # Normalized metric (rotation per second)
            'rotation_rate_per_second': integrated_rotation / total_duration if total_duration > 0 else 0
        }

        self.metrics['goal_c'] = metrics
        logger.info(f"  Total integrated rotation: {integrated_rotation:.1f} deg")
        logger.info(f"  Mean rotation rate: {metrics['mean_rotation_rate']:.2f} deg/s")
        logger.info(f"  Duration: {total_duration:.1f} s")

        return metrics

    def _empty_imu_metrics(self) -> Dict[str, float]:
        """Return empty IMU metrics dictionary."""
        return {
            'total_rotation_deg': 0.0,
            'mean_rotation_rate': 0.0,
            'rotation_rate_std': 0.0,
            'rotation_rate_max': 0.0,
            'gyro_x_mean': 0.0,
            'gyro_x_std': 0.0,
            'gyro_y_mean': 0.0,
            'gyro_y_std': 0.0,
            'gyro_z_mean': 0.0,
            'gyro_z_std': 0.0,
            'total_duration_s': 0.0,
            'n_samples': 0,
            'rotation_rate_per_second': 0.0
        }

    # ==================== GOAL D: Visual Strategy ====================

    def classify_gaze_target(
        self,
        video_path: str = None,
        sample_interval: int = 10
    ) -> Dict[str, float]:
        """
        Goal D: Classify gaze targets as Tool vs Tissue.

        Logic: Novices look at tools; experts look at tissue.
        Uses HSV color thresholding on ROI around gaze point.

        Args:
            video_path: Path to video (overrides constructor)
            sample_interval: Process every Nth gaze sample

        Returns:
            Dictionary with visual strategy metrics:
            - tool_gaze_proportion: Proportion of gaze on tools
            - tissue_gaze_proportion: Proportion of gaze on tissue
            - other_gaze_proportion: Unclassified targets
            - tool_tissue_ratio: Ratio of tool to tissue gaze
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        path = video_path or self.video_path
        if path is None:
            logger.warning("No video path provided for Goal D analysis")
            return self._empty_visual_strategy_metrics()

        logger.info("Classifying gaze targets (Goal D)...")
        logger.info(f"Video path: {path}")

        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {path}")
                return self._empty_visual_strategy_metrics()

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Get valid gaze data with frame indices
            mask = (
                self.data['transformed_gaze_x'].notna() &
                self.data['transformed_gaze_y'].notna() &
                self.data['active_frame_index'].notna()
            )
            valid_data = self.data[mask].iloc[::sample_interval].copy()

            if len(valid_data) < 100:
                logger.warning("Insufficient valid gaze data for visual strategy")
                cap.release()
                return self._empty_visual_strategy_metrics()

            # Classification counters
            tool_count = 0
            tissue_count = 0
            other_count = 0

            # ROI size around gaze point (pixels)
            roi_size = 50

            logger.info(f"Processing {len(valid_data)} gaze samples...")

            for idx, row in valid_data.iterrows():
                frame_idx = int(row['active_frame_index'])
                gaze_x = row['transformed_gaze_x']
                gaze_y = row['transformed_gaze_y']

                # Seek to frame (only if different from current)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    other_count += 1
                    continue

                # Extract ROI around gaze point
                h, w = frame.shape[:2]
                x1 = max(0, int(gaze_x - roi_size))
                y1 = max(0, int(gaze_y - roi_size))
                x2 = min(w, int(gaze_x + roi_size))
                y2 = min(h, int(gaze_y + roi_size))

                if x2 <= x1 or y2 <= y1:
                    other_count += 1
                    continue

                roi = frame[y1:y2, x1:x2]

                # Convert to HSV
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Classify based on color
                classification = self._classify_roi_color(hsv)

                if classification == 'TOOL':
                    tool_count += 1
                elif classification == 'TISSUE':
                    tissue_count += 1
                else:
                    other_count += 1

            cap.release()

            # Calculate proportions
            total = tool_count + tissue_count + other_count

            metrics = {
                'tool_gaze_proportion': tool_count / total if total > 0 else 0,
                'tissue_gaze_proportion': tissue_count / total if total > 0 else 0,
                'other_gaze_proportion': other_count / total if total > 0 else 0,
                'tool_gaze_count': tool_count,
                'tissue_gaze_count': tissue_count,
                'other_gaze_count': other_count,
                'tool_tissue_ratio': tool_count / tissue_count if tissue_count > 0 else float('inf'),
                'total_classified': total
            }

            self.metrics['goal_d'] = metrics
            logger.info(f"  Tool gaze: {metrics['tool_gaze_proportion']*100:.1f}%")
            logger.info(f"  Tissue gaze: {metrics['tissue_gaze_proportion']*100:.1f}%")
            logger.info(f"  Tool/Tissue ratio: {metrics['tool_tissue_ratio']:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"Error in visual strategy analysis: {e}")
            return self._empty_visual_strategy_metrics()

    def _classify_roi_color(self, hsv_roi: np.ndarray) -> str:
        """
        Classify ROI as tool or tissue based on HSV color analysis.

        Tool colors: Grey/Silver (low saturation, medium value)
        Tissue colors: Pink/Yellow/Red (specific hue ranges, higher saturation)
        """
        # Calculate mean HSV values
        h_mean = np.mean(hsv_roi[:, :, 0])
        s_mean = np.mean(hsv_roi[:, :, 1])
        v_mean = np.mean(hsv_roi[:, :, 2])

        # Tool detection: Low saturation (grey/metallic)
        # Metallic tools are typically grey with low saturation
        if s_mean < 50 and 50 < v_mean < 200:
            return 'TOOL'

        # Tissue detection: Pink/Red/Yellow tones
        # Pink: H around 0-10 or 170-180, moderate S
        # Yellow: H around 20-40, moderate S
        # Red: H around 0-10 or 170-180, higher S

        # Pink/red tissue
        if (h_mean < 20 or h_mean > 160) and s_mean > 50 and v_mean > 100:
            return 'TISSUE'

        # Yellow tissue (fat)
        if 15 < h_mean < 45 and s_mean > 40 and v_mean > 100:
            return 'TISSUE'

        return 'OTHER'

    def _empty_visual_strategy_metrics(self) -> Dict[str, float]:
        """Return empty visual strategy metrics dictionary."""
        return {
            'tool_gaze_proportion': 0.0,
            'tissue_gaze_proportion': 0.0,
            'other_gaze_proportion': 0.0,
            'tool_gaze_count': 0,
            'tissue_gaze_count': 0,
            'other_gaze_count': 0,
            'tool_tissue_ratio': 0.0,
            'total_classified': 0
        }

    # ==================== Complete Analysis ====================

    def run_complete_analysis(
        self,
        csv_path: str = None,
        video_path: str = None,
        include_goal_d: bool = True
    ) -> Dict[str, Dict]:
        """
        Run complete whole-session analysis (Goals A-D).

        Args:
            csv_path: Path to enhanced gaze CSV
            video_path: Path to video (for Goal D)
            include_goal_d: Whether to run visual strategy analysis

        Returns:
            Dictionary with all metrics organized by goal
        """
        logger.info("=" * 60)
        logger.info("WHOLE-SESSION ANALYSIS")
        logger.info("=" * 60)

        # Load data
        self.load_data(csv_path)

        # Goal A: Oculometric Efficiency
        logger.info("\n--- Goal A: Oculometric Efficiency ---")
        goal_a = self.calculate_global_fixation_rate()

        # Goal B: Cognitive Load
        logger.info("\n--- Goal B: Cognitive Load ---")
        goal_b = self.calculate_luminance_adjusted_pupil_residuals()

        # Goal C: Motor Stability
        logger.info("\n--- Goal C: Motor Stability ---")
        goal_c = self.calculate_integrated_gyro_motion()

        # Goal D: Visual Strategy (optional)
        goal_d = None
        if include_goal_d and (video_path or self.video_path):
            logger.info("\n--- Goal D: Visual Strategy ---")
            goal_d = self.classify_gaze_target(video_path)

        # Compile results
        results = {
            'goal_a_oculometric_efficiency': goal_a,
            'goal_b_cognitive_load': goal_b,
            'goal_c_motor_stability': goal_c,
            'goal_d_visual_strategy': goal_d,
            'recording_duration_s': goal_a.get('total_duration_s', 0)
        }

        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)

        return results

    def generate_summary_report(self) -> str:
        """
        Generate a human-readable summary report of all metrics.

        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 70)
        lines.append("WHOLE-SESSION SURGICAL SKILL ANALYSIS REPORT")
        lines.append("=" * 70)

        if 'goal_a' in self.metrics:
            m = self.metrics['goal_a']
            lines.append("\nGOAL A: OCULOMETRIC EFFICIENCY")
            lines.append("-" * 40)
            lines.append(f"  Fixation Rate:           {m['fixation_rate_hz']:.2f} Hz")
            lines.append(f"  Fixation Count:          {m['fixation_count']}")
            lines.append(f"  Fixation Proportion:     {m['fixation_proportion']*100:.1f}%")
            lines.append(f"  Mean Fixation Duration:  {m['mean_fixation_duration_ms']:.0f} ms")
            lines.append(f"  Saccade Count:           {m['saccade_count']}")
            lines.append(f"  Recording Duration:      {m['total_duration_s']:.1f} s")

        if 'goal_b' in self.metrics:
            m = self.metrics['goal_b']
            lines.append("\nGOAL B: COGNITIVE LOAD (Pupil Analysis)")
            lines.append("-" * 40)
            lines.append(f"  Mean Pupil Residual:     {m['mean_residual']:.4f} mm")
            lines.append(f"  Residual Std Dev:        {m['std_residual']:.4f} mm")
            lines.append(f"  Luminance R²:            {m['regression_r_squared']:.3f}")
            lines.append(f"  Raw Pupil Mean:          {m['raw_pupil_mean']:.2f} mm")
            if m['mean_residual'] > 0:
                lines.append("  Interpretation:          ELEVATED cognitive load")
            else:
                lines.append("  Interpretation:          LOWER cognitive load")

        if 'goal_c' in self.metrics:
            m = self.metrics['goal_c']
            lines.append("\nGOAL C: MOTOR STABILITY (Head Movement)")
            lines.append("-" * 40)
            lines.append(f"  Total Rotation:          {m['total_rotation_deg']:.1f} deg")
            lines.append(f"  Mean Rotation Rate:      {m['mean_rotation_rate']:.2f} deg/s")
            lines.append(f"  Rotation Rate Std:       {m['rotation_rate_std']:.2f} deg/s")
            lines.append(f"  Normalized Rate:         {m['rotation_rate_per_second']:.2f} deg/s")

        if 'goal_d' in self.metrics and self.metrics['goal_d']:
            m = self.metrics['goal_d']
            lines.append("\nGOAL D: VISUAL STRATEGY (Tool vs Tissue)")
            lines.append("-" * 40)
            lines.append(f"  Tool Gaze:               {m['tool_gaze_proportion']*100:.1f}%")
            lines.append(f"  Tissue Gaze:             {m['tissue_gaze_proportion']*100:.1f}%")
            lines.append(f"  Other:                   {m['other_gaze_proportion']*100:.1f}%")
            lines.append(f"  Tool/Tissue Ratio:       {m['tool_tissue_ratio']:.2f}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


def analyze_subject(
    csv_path: str,
    video_path: str = None,
    include_goal_d: bool = False
) -> Dict:
    """
    Convenience function to analyze a single subject.

    Args:
        csv_path: Path to enhanced final_gaze_data.csv
        video_path: Path to scenevideo.mp4 (optional, for Goal D)
        include_goal_d: Whether to run visual strategy analysis

    Returns:
        Dictionary with all analysis results
    """
    analyzer = WholeSessionAnalyzer()
    results = analyzer.run_complete_analysis(
        csv_path,
        video_path,
        include_goal_d=include_goal_d
    )
    return results
