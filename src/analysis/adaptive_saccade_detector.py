"""
Adaptive Saccade Detection Module

Implements MAD (Median Absolute Deviation) adaptive threshold saccade detection
based on: "MAD saccade: statistically robust saccade threshold estimation"
(Voloh et al., 2020) - https://pmc.ncbi.nlm.nih.gov/articles/PMC7881893/

Key improvements over fixed threshold:
1. Adaptive threshold using MAD-based estimation
2. Amplitude calculation from 3D gaze direction vectors (not velocity integration)
3. Physiological filtering (duration, amplitude, velocity constraints)
4. Main sequence validation

Threshold formula:
    threshold = median(velocity) + lambda * 1.4826 * MAD(velocity)
    where MAD = median(|velocity - median(velocity)|)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy import stats

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SaccadeEvent:
    """Represents a detected saccade with associated metrics."""
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration_ms: float
    amplitude_deg: float
    peak_velocity_deg_s: float
    is_valid: bool = True
    quality_score: float = 1.0
    rejection_reason: Optional[str] = None


class AdaptiveSaccadeDetector:
    """
    Adaptive saccade detector using MAD-based threshold estimation.

    Applies physiological constraints to filter out non-physiological
    eye movements and produces main sequence-compliant saccade detection.
    """

    # Physiological constraints based on literature
    MIN_DURATION_MS = 20.0      # Minimum saccade duration
    MAX_DURATION_MS = 100.0     # Maximum saccade duration
    MIN_AMPLITUDE_DEG = 0.5     # Minimum detectable amplitude
    MAX_AMPLITUDE_DEG = 50.0    # Maximum physiological amplitude
    MAX_PEAK_VELOCITY = 1000.0  # Maximum peak velocity (deg/s)
    MAX_THRESHOLD = 200.0       # Upper bound for adaptive threshold

    # MAD scale factor (converts MAD to standard deviation for normal distribution)
    MAD_SCALE = 1.4826

    def __init__(
        self,
        lambda_factor: float = 6.0,
        max_iterations: int = 10,
        convergence_threshold: float = 1.0
    ):
        """
        Initialize the adaptive saccade detector.

        Args:
            lambda_factor: Multiplier for MAD in threshold calculation (default 6)
            max_iterations: Maximum iterations for threshold convergence
            convergence_threshold: Stop when threshold change < this value (deg/s)
        """
        self.lambda_factor = lambda_factor
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Store computed values
        self.adaptive_threshold = None
        self.threshold_iterations = 0

    def compute_mad_threshold(self, velocities: np.ndarray) -> float:
        """
        Iteratively compute MAD-based adaptive threshold.

        Algorithm:
        1. Start with threshold = infinity
        2. Select velocities below threshold
        3. Compute median and MAD of selected velocities
        4. Update threshold = median + lambda * 1.4826 * MAD
        5. Repeat until convergence (<1 deg/s change) or max iterations
        6. Apply upper bound (MAX_THRESHOLD)

        Args:
            velocities: Array of angular velocities (deg/s)

        Returns:
            Adaptive threshold in deg/s
        """
        # Remove NaN values
        valid_velocities = velocities[~np.isnan(velocities)]

        if len(valid_velocities) < 10:
            logger.warning("Insufficient velocity data for MAD threshold")
            return self.MAX_THRESHOLD

        # Initialize with all velocities (threshold = infinity)
        threshold = np.inf
        prev_threshold = np.inf

        for iteration in range(self.max_iterations):
            # Select velocities below current threshold
            below_threshold = valid_velocities[valid_velocities < threshold]

            if len(below_threshold) < 10:
                logger.warning(f"Iteration {iteration}: Too few samples below threshold")
                break

            # Compute median velocity
            median_vel = np.median(below_threshold)

            # Compute MAD (Median Absolute Deviation)
            mad = np.median(np.abs(below_threshold - median_vel))

            # Update threshold
            prev_threshold = threshold
            threshold = median_vel + self.lambda_factor * self.MAD_SCALE * mad

            # Check for convergence
            change = abs(threshold - prev_threshold) if prev_threshold != np.inf else np.inf

            logger.debug(f"Iteration {iteration}: threshold={threshold:.2f}, "
                        f"median={median_vel:.2f}, MAD={mad:.2f}, change={change:.2f}")

            if change < self.convergence_threshold:
                logger.info(f"MAD threshold converged at iteration {iteration}")
                break

        self.threshold_iterations = iteration + 1

        # Apply upper bound
        threshold = min(threshold, self.MAX_THRESHOLD)

        self.adaptive_threshold = threshold
        logger.info(f"Adaptive threshold: {threshold:.2f} deg/s "
                   f"(converged in {self.threshold_iterations} iterations)")

        return threshold

    def detect_saccades(
        self,
        df: pd.DataFrame,
        velocity_col: str = 'angular_velocity_deg_s',
        timestamp_col: str = 'gaze_timestamp'
    ) -> List[SaccadeEvent]:
        """
        Detect saccades using adaptive threshold and physiological filtering.

        Args:
            df: DataFrame with gaze data
            velocity_col: Column name for angular velocity
            timestamp_col: Column name for timestamps

        Returns:
            List of SaccadeEvent objects
        """
        if velocity_col not in df.columns:
            logger.error(f"Velocity column '{velocity_col}' not found")
            return []

        # Get valid velocity data
        mask = df[velocity_col].notna()
        valid_df = df[mask].copy()

        if len(valid_df) < 100:
            logger.warning("Insufficient data for saccade detection")
            return []

        velocities = valid_df[velocity_col].values
        timestamps = valid_df[timestamp_col].values

        # Compute adaptive threshold
        threshold = self.compute_mad_threshold(velocities)

        # Detect saccade events (velocity above threshold)
        saccades = []
        in_saccade = False
        saccade_start_idx = 0

        for i, vel in enumerate(velocities):
            if vel >= threshold:
                if not in_saccade:
                    in_saccade = True
                    saccade_start_idx = i
            else:
                if in_saccade:
                    in_saccade = False
                    saccade = self._create_saccade_event(
                        valid_df, saccade_start_idx, i,
                        velocities, timestamps
                    )
                    if saccade is not None:
                        saccades.append(saccade)

        # Handle saccade extending to end
        if in_saccade:
            saccade = self._create_saccade_event(
                valid_df, saccade_start_idx, len(velocities),
                velocities, timestamps
            )
            if saccade is not None:
                saccades.append(saccade)

        logger.info(f"Detected {len(saccades)} raw saccade events")

        # Apply physiological filtering
        filtered_saccades = self.filter_physiological(saccades)
        logger.info(f"After physiological filtering: {len(filtered_saccades)} saccades")

        # Validate main sequence
        validated_saccades = self.validate_main_sequence(filtered_saccades)
        valid_count = sum(1 for s in validated_saccades if s.is_valid)
        logger.info(f"Main sequence validation: {valid_count}/{len(validated_saccades)} valid")

        return validated_saccades

    def _create_saccade_event(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> Optional[SaccadeEvent]:
        """Create a SaccadeEvent from detected indices."""
        if end_idx <= start_idx:
            return None

        # Extract saccade data
        saccade_velocities = velocities[start_idx:end_idx]
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx - 1] if end_idx > start_idx else start_time

        # Calculate metrics
        duration_ms = (end_time - start_time) * 1000
        peak_velocity = np.max(saccade_velocities)

        # Calculate amplitude from gaze direction vectors if available
        amplitude = self.calculate_amplitude_from_vectors(df, start_idx, end_idx)

        if amplitude is None:
            # Fallback: estimate from velocity (less accurate)
            amplitude = self._estimate_amplitude_from_velocity(
                saccade_velocities, timestamps[start_idx:end_idx]
            )

        return SaccadeEvent(
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            amplitude_deg=amplitude,
            peak_velocity_deg_s=peak_velocity
        )

    def calculate_amplitude_from_vectors(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> Optional[float]:
        """
        Calculate saccade amplitude from 3D gaze direction vectors.

        Uses the formula: amplitude = arccos(dot(gaze_start, gaze_end))
        This gives the true angular displacement, not path length.

        Args:
            df: DataFrame with gaze direction columns
            start_idx: Start index of saccade
            end_idx: End index of saccade

        Returns:
            Amplitude in degrees, or None if vectors not available
        """
        # Check if gaze direction columns exist
        gaze_cols = ['gaze_direction_x', 'gaze_direction_y', 'gaze_direction_z']
        if not all(col in df.columns for col in gaze_cols):
            return None

        try:
            # Get start and end gaze direction vectors
            start_row = df.iloc[start_idx]
            end_row = df.iloc[end_idx - 1] if end_idx > start_idx else start_row

            gaze_start = np.array([
                start_row['gaze_direction_x'],
                start_row['gaze_direction_y'],
                start_row['gaze_direction_z']
            ])

            gaze_end = np.array([
                end_row['gaze_direction_x'],
                end_row['gaze_direction_y'],
                end_row['gaze_direction_z']
            ])

            # Check for NaN values
            if np.any(np.isnan(gaze_start)) or np.any(np.isnan(gaze_end)):
                return None

            # Normalize vectors
            norm_start = np.linalg.norm(gaze_start)
            norm_end = np.linalg.norm(gaze_end)

            if norm_start < 1e-6 or norm_end < 1e-6:
                return None

            gaze_start = gaze_start / norm_start
            gaze_end = gaze_end / norm_end

            # Calculate angle between vectors
            dot_product = np.dot(gaze_start, gaze_end)
            dot_product = np.clip(dot_product, -1.0, 1.0)

            amplitude_rad = np.arccos(dot_product)
            amplitude_deg = np.degrees(amplitude_rad)

            return amplitude_deg

        except Exception as e:
            logger.debug(f"Error calculating amplitude from vectors: {e}")
            return None

    def _estimate_amplitude_from_velocity(
        self,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> float:
        """
        Fallback amplitude estimation from velocity.

        Uses trapezoidal integration but applies a correction factor
        since this tends to overestimate due to path length vs displacement.

        Args:
            velocities: Array of velocities during saccade
            timestamps: Corresponding timestamps

        Returns:
            Estimated amplitude in degrees
        """
        if len(velocities) < 2:
            return 0.0

        dt = np.diff(timestamps)

        # Trapezoidal integration
        path_length = np.sum((velocities[:-1] + velocities[1:]) / 2 * dt)

        # Apply correction factor (empirical: displacement ~ 0.7 * path_length for saccades)
        # This accounts for the curved trajectory
        amplitude = path_length * 0.7

        return amplitude

    def filter_physiological(
        self,
        saccades: List[SaccadeEvent]
    ) -> List[SaccadeEvent]:
        """
        Filter saccades based on physiological constraints.

        Removes saccades that fall outside normal physiological ranges:
        - Duration: 20-100 ms
        - Amplitude: 0.5-50 degrees
        - Peak velocity: <1000 deg/s

        Args:
            saccades: List of detected saccade events

        Returns:
            Filtered list of saccade events
        """
        filtered = []
        rejection_counts = {
            'duration_short': 0,
            'duration_long': 0,
            'amplitude_small': 0,
            'amplitude_large': 0,
            'velocity_high': 0
        }

        for saccade in saccades:
            # Check duration
            if saccade.duration_ms < self.MIN_DURATION_MS:
                saccade.is_valid = False
                saccade.rejection_reason = 'duration_too_short'
                rejection_counts['duration_short'] += 1
                continue

            if saccade.duration_ms > self.MAX_DURATION_MS:
                saccade.is_valid = False
                saccade.rejection_reason = 'duration_too_long'
                rejection_counts['duration_long'] += 1
                continue

            # Check amplitude
            if saccade.amplitude_deg < self.MIN_AMPLITUDE_DEG:
                saccade.is_valid = False
                saccade.rejection_reason = 'amplitude_too_small'
                rejection_counts['amplitude_small'] += 1
                continue

            if saccade.amplitude_deg > self.MAX_AMPLITUDE_DEG:
                saccade.is_valid = False
                saccade.rejection_reason = 'amplitude_too_large'
                rejection_counts['amplitude_large'] += 1
                continue

            # Check peak velocity
            if saccade.peak_velocity_deg_s > self.MAX_PEAK_VELOCITY:
                saccade.is_valid = False
                saccade.rejection_reason = 'velocity_too_high'
                rejection_counts['velocity_high'] += 1
                continue

            filtered.append(saccade)

        # Log rejection statistics
        total_rejected = sum(rejection_counts.values())
        if total_rejected > 0:
            logger.info(f"Physiological filtering rejected {total_rejected} saccades:")
            for reason, count in rejection_counts.items():
                if count > 0:
                    logger.info(f"  - {reason}: {count}")

        return filtered

    def validate_main_sequence(
        self,
        saccades: List[SaccadeEvent],
        outlier_threshold_sd: float = 3.0
    ) -> List[SaccadeEvent]:
        """
        Validate saccades against main sequence relationship.

        Fits the main sequence: V_peak = k * A^n
        Flags saccades that deviate >3 SD from the fit as low quality.

        Args:
            saccades: List of saccade events
            outlier_threshold_sd: Number of SDs for outlier detection

        Returns:
            List of saccades with quality scores updated
        """
        if len(saccades) < 10:
            # Not enough saccades for reliable fit
            return saccades

        # Extract amplitudes and peak velocities
        amplitudes = np.array([s.amplitude_deg for s in saccades])
        peak_velocities = np.array([s.peak_velocity_deg_s for s in saccades])

        # Filter out zeros and negatives for log fit
        valid_mask = (amplitudes > 0) & (peak_velocities > 0)
        if valid_mask.sum() < 10:
            return saccades

        log_amp = np.log(amplitudes[valid_mask])
        log_vel = np.log(peak_velocities[valid_mask])

        try:
            # Fit linear model in log space: log(V) = log(k) + n*log(A)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_amp, log_vel)

            # Calculate residuals
            predicted_log_vel = intercept + slope * log_amp
            residuals = log_vel - predicted_log_vel
            residual_std = np.std(residuals)

            logger.info(f"Main sequence fit: V = {np.exp(intercept):.1f} * A^{slope:.2f}, "
                       f"R² = {r_value**2:.3f}")

            # Update quality scores based on residual
            valid_idx = 0
            for i, saccade in enumerate(saccades):
                if valid_mask[i] if i < len(valid_mask) else False:
                    residual_sd = abs(residuals[valid_idx]) / residual_std if residual_std > 0 else 0

                    # Quality score: 1.0 at residual=0, decreasing with distance
                    quality = max(0, 1 - residual_sd / outlier_threshold_sd)
                    saccade.quality_score = quality

                    if residual_sd > outlier_threshold_sd:
                        saccade.is_valid = False
                        saccade.rejection_reason = 'main_sequence_outlier'

                    valid_idx += 1

        except Exception as e:
            logger.warning(f"Main sequence fit failed: {e}")

        return saccades

    def get_summary_statistics(
        self,
        saccades: List[SaccadeEvent]
    ) -> dict:
        """
        Calculate summary statistics for detected saccades.

        Args:
            saccades: List of saccade events

        Returns:
            Dictionary of summary statistics
        """
        valid_saccades = [s for s in saccades if s.is_valid]

        if not valid_saccades:
            return {
                'total_detected': len(saccades),
                'valid_count': 0,
                'adaptive_threshold_deg_s': self.adaptive_threshold,
                'threshold_iterations': self.threshold_iterations
            }

        amplitudes = np.array([s.amplitude_deg for s in valid_saccades])
        velocities = np.array([s.peak_velocity_deg_s for s in valid_saccades])
        durations = np.array([s.duration_ms for s in valid_saccades])

        # Calculate main sequence R²
        main_seq_r2 = 0.0
        if len(amplitudes) >= 10:
            log_amp = np.log(amplitudes[amplitudes > 0])
            log_vel = np.log(velocities[amplitudes > 0])
            if len(log_amp) >= 10:
                _, _, r_value, _, _ = stats.linregress(log_amp, log_vel)
                main_seq_r2 = r_value ** 2

        return {
            'total_detected': len(saccades),
            'valid_count': len(valid_saccades),
            'rejected_count': len(saccades) - len(valid_saccades),
            'adaptive_threshold_deg_s': self.adaptive_threshold,
            'threshold_iterations': self.threshold_iterations,
            'amplitude_mean_deg': float(np.mean(amplitudes)),
            'amplitude_std_deg': float(np.std(amplitudes)),
            'amplitude_max_deg': float(np.max(amplitudes)),
            'peak_velocity_mean_deg_s': float(np.mean(velocities)),
            'peak_velocity_std_deg_s': float(np.std(velocities)),
            'peak_velocity_max_deg_s': float(np.max(velocities)),
            'duration_mean_ms': float(np.mean(durations)),
            'duration_std_ms': float(np.std(durations)),
            'main_sequence_r_squared': main_seq_r2
        }
