"""
Whole-Session Analysis Module for Surgical Skill Assessment

This module performs the high-level scientific analysis. It calculates metrics for
the four key goals of the project:

- **Goal A: Oculometric Efficiency**: Is the surgeon searching efficiently?
- **Goal B: Cognitive Load**: Is the surgeon mentally stressed?
- **Goal C: Motor Stability**: Is the surgeon's head stable?
- **Goal D: Visual Strategy**: Is the surgeon looking at tools or tissue? (Optional)

These metrics are derived from the enhanced CSV generated in the previous step,
which contains physics-based data (angular velocity, pupil diameter, IMU).
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
from .adaptive_saccade_detector import AdaptiveSaccadeDetector, SaccadeEvent
from .pupil_luminance_kernel import PupilLuminanceKernel

logger = get_logger(__name__)


class WholeSessionAnalyzer:
    """
    Analyzes an entire surgical session to produce a skill report.

    Since we often don't have timestamps for specific sub-tasks, this analyzer
    treats the whole recording as one continuous performance block.
    """

    # I-VT velocity thresholds (degrees/second)
    # Velocity below this is considered a "Fixation" (eye is still)
    FIXATION_THRESHOLD = 30.0

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

        # Raw data storage for visualizations
        self.fixation_durations = []      # ms
        self.saccade_amplitudes = []      # degrees
        self.saccade_peak_velocities = [] # deg/s
        self.saccade_events = []          # List[SaccadeEvent]
        self.pupil_residuals = None       # np.ndarray
        self.pupil_timestamps = None      # np.ndarray
        self.gyro_magnitude = None        # np.ndarray
        self.gyro_timestamps = None       # np.ndarray
        self.gyro_x = None                # np.ndarray (pitch)
        self.gyro_y = None                # np.ndarray (yaw)

        # Saccade detection (Goal A)
        self.saccade_detector = AdaptiveSaccadeDetector()
        self.adaptive_threshold = None    # Computed saccade threshold

        # Pupil-luminance kernel analysis (Goal B)
        self.kernel_model = None          # PupilLuminanceKernel instance
        self.convolved_luminance = None   # np.ndarray for visualization
        self.pupil_predicted = None       # np.ndarray (from convolved regression)

    def load_data(self, csv_path: str = None) -> pd.DataFrame:
        """
        Load the enhanced gaze data CSV.
        """
        path = csv_path or self.csv_path
        if path is None:
            raise ValueError("No CSV path provided")

        logger.info(f"Loading gaze data from: {path}")
        self.data = pd.read_csv(path)

        logger.info(f"Loaded {len(self.data)} records")

        # Validate required columns
        required = ['gaze_timestamp', 'angular_velocity_deg_s']
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")

        return self.data

    # ==================== GOAL A: Oculometric Efficiency ====================

    def calculate_global_fixation_rate(self) -> Dict[str, float]:
        """
        Goal A: How efficient is the visual search?

        Metric: **Fixation Rate (Hz)**
        - High rate = frequent, short stops. Often indicates confusion or searching (Novice).
        - Low rate = fewer, longer stops. Indicates processing and planning (Expert).

        Returns:
            Dictionary with fixation metrics: rates, counts, proportions.
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

        velocities = valid_data['angular_velocity_deg_s'].values
        timestamps = valid_data['gaze_timestamp'].values

        # 1. Detect Saccades using Adaptive MAD threshold
        # This fills self.saccade_events and calculates self.adaptive_threshold
        self._calculate_saccade_events(None, timestamps, velocities, valid_data)

        # 2. Determine Classification Thresholds
        saccade_threshold = self.adaptive_threshold if self.adaptive_threshold else 200.0

        # 3. Classify every sample
        states = np.empty(len(velocities), dtype=object)
        states[velocities < self.FIXATION_THRESHOLD] = 'FIXATION'
        states[velocities >= saccade_threshold] = 'SACCADE'
        states[(velocities >= self.FIXATION_THRESHOLD) &
               (velocities < saccade_threshold)] = 'OTHER'

        # 4. Count Fixation Events (transitions into fixation state)
        fixation_starts = (states == 'FIXATION') & (np.roll(states, 1) != 'FIXATION')
        fixation_starts[0] = states[0] == 'FIXATION'  # Handle first sample
        fixation_count = fixation_starts.sum()

        # Use saccade count from adaptive detector (more accurate with physiological filtering)
        valid_saccade_count = len([s for s in self.saccade_events if s.is_valid])

        # Calculate total duration
        total_duration = timestamps[-1] - timestamps[0]

        # Calculate fixation rate (Hz)
        fixation_rate_hz = fixation_count / total_duration if total_duration > 0 else 0

        # Calculate proportions of time spent in each state
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

        # Store raw fixation durations for visualization (in ms)
        self.fixation_durations = [d * 1000 for d in fixation_durations]

        # Get saccade statistics from adaptive detector
        saccade_stats = self.saccade_detector.get_summary_statistics(self.saccade_events)

        metrics = {
            'fixation_rate_hz': fixation_rate_hz,
            'fixation_count': int(fixation_count),
            'fixation_proportion': fixation_proportion,
            'mean_fixation_duration_ms': mean_fixation_duration_ms,
            'saccade_count': valid_saccade_count,
            'saccade_proportion': saccade_proportion,
            'total_duration_s': total_duration,
            'total_samples': total_samples,
            # New adaptive saccade detection metrics
            'adaptive_threshold_deg_s': self.adaptive_threshold,
            'saccade_amplitude_mean_deg': saccade_stats.get('amplitude_mean_deg', 0),
            'saccade_amplitude_max_deg': saccade_stats.get('amplitude_max_deg', 0),
            'saccade_peak_velocity_mean_deg_s': saccade_stats.get('peak_velocity_mean_deg_s', 0),
            'saccade_peak_velocity_max_deg_s': saccade_stats.get('peak_velocity_max_deg_s', 0),
            'main_sequence_r_squared': saccade_stats.get('main_sequence_r_squared', 0),
        }

        self.metrics['goal_a'] = metrics
        logger.info(f"  Fixation rate: {fixation_rate_hz:.2f} Hz")
        logger.info(f"  Valid saccade count: {valid_saccade_count} (adaptive threshold: {self.adaptive_threshold:.1f} deg/s)")

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

        if in_event:
            durations.append(timestamps[-1] - event_start)

        return durations

    def _calculate_saccade_events(
        self,
        states: np.ndarray,
        timestamps: np.ndarray,
        velocities: np.ndarray,
        valid_data: pd.DataFrame
    ) -> None:
        """
        Extract saccade events using adaptive MAD-based threshold detection.
        Uses the AdaptiveSaccadeDetector class.
        """
        # Use the adaptive saccade detector
        saccade_events = self.saccade_detector.detect_saccades(valid_data)

        # Store the adaptive threshold found by the detector
        self.adaptive_threshold = self.saccade_detector.adaptive_threshold

        # Extract valid saccades for visualization
        valid_events = [s for s in saccade_events if s.is_valid]

        self.saccade_events = saccade_events
        self.saccade_amplitudes = [s.amplitude_deg for s in valid_events]
        self.saccade_peak_velocities = [s.peak_velocity_deg_s for s in valid_events]

        # Get summary statistics
        stats = self.saccade_detector.get_summary_statistics(saccade_events)
        logger.info(f"  Adaptive threshold: {self.adaptive_threshold:.1f} deg/s")

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
            'total_samples': 0,
            'adaptive_threshold_deg_s': 0.0,
            'saccade_amplitude_mean_deg': 0.0,
            'saccade_amplitude_max_deg': 0.0,
            'saccade_peak_velocity_mean_deg_s': 0.0,
            'saccade_peak_velocity_max_deg_s': 0.0,
            'main_sequence_r_squared': 0.0,
        }

    # ==================== GOAL B: Cognitive Load ====================

    def calculate_luminance_adjusted_pupil_residuals(self) -> Dict[str, float]:
        """
        Goal B: How hard is the surgeon thinking?

        The pupil dilates for two reasons:
        1. Darkness (Pupillary Light Reflex - PLR)
        2. Mental Effort (Cognitive Load)

        To measure #2, we must subtract #1.

        We build a mathematical model (Kernel Regression) that predicts
        how the pupil *should* react to the light seen in the video.
        Any dilation *beyond* that prediction is assumed to be cognitive load.
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
        timestamps = valid_data['gaze_timestamp'].values

        # Estimate sampling rate from timestamps
        dt = np.median(np.diff(timestamps))
        sampling_rate = 1.0 / dt if dt > 0 else 90.0

        # === TEMPORAL KERNEL FITTING ===
        self.kernel_model = PupilLuminanceKernel(sampling_rate_hz=sampling_rate)

        # Fit kernel parameters (t_max, n) to this subject's data
        # This customizes the light reflex model to the individual
        kernel_params = self.kernel_model.fit_to_subject(pupil, luminance)

        # Get full regression results (predict pupil from light)
        kernel_results = self.kernel_model.fit_regression(pupil, luminance)

        # Store data for visualization
        self.pupil_timestamps = timestamps
        self.pupil_residuals = kernel_results['residuals_convolved']
        self.convolved_luminance = kernel_results['convolved_luminance']
        self.pupil_predicted = kernel_results['predicted_convolved']

        metrics = {
            # Instantaneous regression (baseline for comparison)
            'regression_r_squared': kernel_results['r_squared_instantaneous'],
            'regression_slope': kernel_results['slope_instantaneous'],
            'regression_intercept': kernel_results['intercept_instantaneous'],
            'regression_p_value': kernel_results['p_value_instantaneous'],

            # Convolved regression (with fitted temporal kernel - THE GOOD MODEL)
            'regression_r_squared_convolved': kernel_results['r_squared_convolved'],
            'regression_slope_convolved': kernel_results['slope_convolved'],
            'regression_intercept_convolved': kernel_results['intercept_convolved'],
            'regression_p_value_convolved': kernel_results['p_value_convolved'],

            # Residuals (High Residuals = High Cognitive Load)
            'mean_residual': kernel_results['residual_mean_convolved'],
            'std_residual': kernel_results['residual_std_convolved'],

            # Kernel parameters (Physiological differences)
            'kernel_t_max_ms': kernel_results['kernel_t_max_ms'],
            'kernel_n': kernel_results['kernel_n'],
            'kernel_is_fitted': kernel_results['kernel_is_fitted'],

            # Improvement metrics
            'r_squared_improvement': kernel_results['r_squared_improvement'],
            'r_squared_improvement_pct': kernel_results['r_squared_improvement_pct'],

            # Raw data statistics
            'raw_pupil_mean': np.mean(pupil),
            'raw_pupil_std': np.std(pupil),
            'luminance_mean': np.mean(luminance),
            'luminance_std': np.std(luminance),
            'n_samples': kernel_results['n_valid_samples']
        }

        self.metrics['goal_b'] = metrics
        logger.info(f"  Convolved R²:     {metrics['regression_r_squared_convolved']:.4f}")

        return metrics

    def _empty_pupil_metrics(self) -> Dict[str, float]:
        """Return empty pupil metrics dictionary."""
        return {
            'regression_r_squared': 0.0,
            'regression_slope': 0.0,
            'regression_intercept': 0.0,
            'regression_p_value': 1.0,
            'regression_r_squared_convolved': 0.0,
            'regression_slope_convolved': 0.0,
            'regression_intercept_convolved': 0.0,
            'regression_p_value_convolved': 1.0,
            'mean_residual': 0.0,
            'std_residual': 0.0,
            'kernel_t_max_ms': 512.0,
            'kernel_n': 10.1,
            'kernel_is_fitted': False,
            'r_squared_improvement': 0.0,
            'r_squared_improvement_pct': 0.0,
            'raw_pupil_mean': 0.0,
            'raw_pupil_std': 0.0,
            'luminance_mean': 0.0,
            'luminance_std': 0.0,
            'n_samples': 0
        }

    # ==================== GOAL C: Motor Stability ====================

    def calculate_integrated_gyro_motion(self) -> Dict[str, float]:
        """
        Goal C: Is the head stable?

        We sum up all the rotation measured by the headset's gyroscope.

        Metric: **Integrated Rotation (degrees)**
        - Lower value = Head was steady (Expert focus).
        - Higher value = Lots of looking around (Novice searching).
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
        # Euclidean norm of the rotation vector
        rotation_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

        # Store raw data for visualization
        self.gyro_magnitude = rotation_magnitude
        self.gyro_timestamps = timestamps
        self.gyro_x = gyro_x
        self.gyro_y = gyro_y

        # Integrate rotation over time (Area under the curve)
        dt = np.diff(timestamps)
        dt = np.clip(dt, 0, 0.1)  # Clip outliers

        # Total integrated rotation
        integrated_rotation = np.sum(rotation_magnitude[:-1] * dt)

        # Duration
        total_duration = timestamps[-1] - timestamps[0]

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
            'rotation_rate_per_second': integrated_rotation / total_duration if total_duration > 0 else 0
        }

        self.metrics['goal_c'] = metrics
        logger.info(f"  Total integrated rotation: {integrated_rotation:.1f} deg")

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
        Goal D: What is the surgeon looking at?

        We look at the pixel color at the gaze location.
        - **Tools**: Usually silver/grey (Low color saturation).
        - **Tissue**: Pink/Red/Yellow (High color saturation).

        Metric: **Tool/Tissue Ratio**
        - Experts look at tissue (low ratio).
        - Novices track their tools (high ratio).
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        path = video_path or self.video_path
        if path is None:
            logger.warning("No video path provided for Goal D analysis")
            return self._empty_visual_strategy_metrics()

        logger.info("Classifying gaze targets (Goal D)...")

        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {path}")
                return self._empty_visual_strategy_metrics()

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

            for idx, row in valid_data.iterrows():
                frame_idx = int(row['active_frame_index'])
                gaze_x = row['transformed_gaze_x']
                gaze_y = row['transformed_gaze_y']

                # Seek to frame
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

                # Convert to HSV (Hue Saturation Value) color space
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
        if s_mean < 50 and 50 < v_mean < 200:
            return 'TOOL'

        # Tissue detection: Pink/Red/Yellow tones
        # Pink: H around 0-10 or 170-180
        # Yellow: H around 20-40
        if (h_mean < 20 or h_mean > 160) and s_mean > 50 and v_mean > 100:
            return 'TISSUE'
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
            lines.append(f"  Recording Duration:      {m['total_duration_s']:.1f} s")

        if 'goal_b' in self.metrics:
            m = self.metrics['goal_b']
            lines.append("\nGOAL B: COGNITIVE LOAD (Pupil Analysis)")
            lines.append("-" * 40)
            lines.append("  Pupil-Luminance Regression:")
            lines.append(f"    Convolved R²:          {m.get('regression_r_squared_convolved', 0):.4f}")
            lines.append("")
            lines.append("  Cognitive Load Metrics:")
            lines.append(f"    Residual Std Dev:      {m['std_residual']:.4f} mm")

        if 'goal_c' in self.metrics:
            m = self.metrics['goal_c']
            lines.append("\nGOAL C: MOTOR STABILITY (Head Movement)")
            lines.append("-" * 40)
            lines.append(f"  Total Rotation:          {m['total_rotation_deg']:.1f} deg")
            lines.append(f"  Mean Rotation Rate:      {m['mean_rotation_rate']:.2f} deg/s")

        if 'goal_d' in self.metrics and self.metrics['goal_d']:
            m = self.metrics['goal_d']
            lines.append("\nGOAL D: VISUAL STRATEGY (Tool vs Tissue)")
            lines.append("-" * 40)
            lines.append(f"  Tool Gaze:               {m['tool_gaze_proportion']*100:.1f}%")
            lines.append(f"  Tissue Gaze:             {m['tissue_gaze_proportion']*100:.1f}%")
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
    """
    analyzer = WholeSessionAnalyzer()
    results = analyzer.run_complete_analysis(
        csv_path,
        video_path,
        include_goal_d=include_goal_d
    )
    return results
