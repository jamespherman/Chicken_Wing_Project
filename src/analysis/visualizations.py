"""
Visualization Module for Surgical Skill Assessment

Implements 4 clinical visualizations based on physics-based gaze data:
1. Cognitive Fingerprint (Violin Plot) - Fixation duration distribution
2. Main Sequence (Scatter + Regression) - Saccade amplitude vs peak velocity
3. Stress Timeline (Stacked Time-Series) - Pupil residuals and gyro magnitude
4. Stability Radar (Polar Histogram) - Head movement direction distribution
"""

# Import scipy BEFORE matplotlib/seaborn to avoid BLAS threading deadlock
from scipy import stats
from scipy.optimize import curve_fit

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..logging_config import get_logger

logger = get_logger(__name__)


class GazeVisualizer:
    """
    Creates clinical visualizations for surgical skill assessment.

    Generates four visualization types from WholeSessionAnalyzer data:
    - Cognitive Fingerprint: Violin plot of fixation durations
    - Main Sequence: Saccade amplitude vs velocity scatter plot
    - Stress Timeline: Pupil residuals and head motion over time
    - Stability Radar: Polar histogram of head movement directions
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the visualizer with configuration.

        Args:
            config: Configuration dictionary with optional keys:
                - figure_size: Tuple (width, height) in inches, default (12, 8)
                - dpi: Output resolution, default 300
                - rolling_window_s: Rolling average window in seconds, default 5
        """
        config = config or {}
        self.figure_size = tuple(config.get('figure_size', (12, 8)))
        self.dpi = config.get('dpi', 300)
        self.rolling_window_s = config.get('rolling_window_s', 5)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')

    def create_all_visualizations(
        self,
        analyzer,
        output_dir: str,
        subject_name: str
    ) -> Dict[str, str]:
        """
        Create all four visualizations from analyzer data.

        Args:
            analyzer: WholeSessionAnalyzer instance with computed metrics
            output_dir: Directory to save visualizations
            subject_name: Subject identifier for filenames

        Returns:
            Dictionary mapping visualization type to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created = {}
        errors = []

        # 1. Cognitive Fingerprint (Fixation Duration Distribution)
        if analyzer.fixation_durations:
            try:
                path = output_dir / f"{subject_name}_viz_cognitive_fingerprint.png"
                self.create_cognitive_fingerprint(
                    analyzer.fixation_durations,
                    subject_name,
                    str(path)
                )
                created['cognitive_fingerprint'] = str(path)
                logger.info(f"  Created cognitive fingerprint: {path.name}")
            except Exception as e:
                errors.append(f"cognitive_fingerprint: {e}")
                logger.warning(f"  Failed to create cognitive fingerprint: {e}")
        else:
            logger.warning("  Skipping cognitive fingerprint: no fixation data")

        # 2. Main Sequence (Saccade Amplitude vs Velocity)
        if analyzer.saccade_amplitudes and analyzer.saccade_peak_velocities:
            try:
                path = output_dir / f"{subject_name}_viz_main_sequence.png"

                # Get adaptive detection statistics if available
                adaptive_threshold = getattr(analyzer, 'adaptive_threshold', None)
                saccade_events = getattr(analyzer, 'saccade_events', [])
                total_detected = len(saccade_events) if saccade_events else None
                rejected_count = sum(1 for s in saccade_events if not s.is_valid) if saccade_events else None

                self.create_main_sequence(
                    analyzer.saccade_amplitudes,
                    analyzer.saccade_peak_velocities,
                    subject_name,
                    str(path),
                    adaptive_threshold=adaptive_threshold,
                    total_detected=total_detected,
                    rejected_count=rejected_count
                )
                created['main_sequence'] = str(path)
                logger.info(f"  Created main sequence: {path.name}")
            except Exception as e:
                errors.append(f"main_sequence: {e}")
                logger.warning(f"  Failed to create main sequence: {e}")
        else:
            logger.warning("  Skipping main sequence: no saccade data")

        # 3. Stress Timeline (Pupil + Gyro Time Series)
        has_pupil = analyzer.pupil_timestamps is not None and analyzer.pupil_residuals is not None
        has_gyro = analyzer.gyro_timestamps is not None and analyzer.gyro_magnitude is not None
        if has_pupil or has_gyro:
            try:
                path = output_dir / f"{subject_name}_viz_stress_timeline.png"
                self.create_stress_timeline(
                    analyzer.pupil_timestamps,
                    analyzer.pupil_residuals,
                    analyzer.gyro_timestamps,
                    analyzer.gyro_magnitude,
                    subject_name,
                    str(path)
                )
                created['stress_timeline'] = str(path)
                logger.info(f"  Created stress timeline: {path.name}")
            except Exception as e:
                errors.append(f"stress_timeline: {e}")
                logger.warning(f"  Failed to create stress timeline: {e}")
        else:
            logger.warning("  Skipping stress timeline: no pupil/gyro data")

        # 4. Stability Radar (Head Movement Direction)
        if analyzer.gyro_x is not None and analyzer.gyro_y is not None:
            try:
                path = output_dir / f"{subject_name}_viz_stability_radar.png"
                self.create_stability_radar(
                    analyzer.gyro_x,
                    analyzer.gyro_y,
                    subject_name,
                    str(path)
                )
                created['stability_radar'] = str(path)
                logger.info(f"  Created stability radar: {path.name}")
            except Exception as e:
                errors.append(f"stability_radar: {e}")
                logger.warning(f"  Failed to create stability radar: {e}")
        else:
            logger.warning("  Skipping stability radar: no gyro data")

        # 5. Kernel Fit (Pupil-Luminance Temporal Analysis)
        has_kernel = (hasattr(analyzer, 'kernel_model') and
                      analyzer.kernel_model is not None and
                      hasattr(analyzer, 'convolved_luminance') and
                      analyzer.convolved_luminance is not None)
        if has_kernel:
            try:
                path = output_dir / f"{subject_name}_viz_kernel_fit.png"
                # Get pupil and luminance data
                mask = (
                    analyzer.data['pupil_diameter_avg'].notna() &
                    analyzer.data['frame_luminance'].notna()
                )
                valid_data = analyzer.data[mask]
                pupil = valid_data['pupil_diameter_avg'].values
                luminance = valid_data['frame_luminance'].values
                timestamps = valid_data['gaze_timestamp'].values

                # Get metrics from analyzer
                metrics = analyzer.metrics.get('goal_b', {})

                self.create_kernel_fit_plot(
                    analyzer.kernel_model,
                    pupil,
                    luminance,
                    timestamps,
                    analyzer.convolved_luminance,
                    metrics,
                    subject_name,
                    str(path)
                )
                created['kernel_fit'] = str(path)
                logger.info(f"  Created kernel fit plot: {path.name}")
            except Exception as e:
                errors.append(f"kernel_fit: {e}")
                logger.warning(f"  Failed to create kernel fit plot: {e}")
        else:
            logger.warning("  Skipping kernel fit: no kernel model available")

        return {
            'created': created,
            'errors': errors,
            'success': len(errors) == 0
        }

    def create_cognitive_fingerprint(
        self,
        fixation_durations: List[float],
        subject_name: str,
        output_path: str
    ) -> None:
        """
        Create violin plot of fixation duration distribution.

        This visualization reveals the surgeon's attention pattern:
        - Short fixations (<150ms): Rapid visual search (novice pattern)
        - Medium fixations (150-400ms): Optimal information gathering
        - Long fixations (>400ms): Extended processing or hesitation

        Args:
            fixation_durations: List of fixation durations in milliseconds
            subject_name: Subject identifier for title
            output_path: Output file path
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Filter extreme outliers for better visualization (keep 1st-99th percentile)
        durations = np.array(fixation_durations)
        p1, p99 = np.percentile(durations, [1, 99])
        filtered = durations[(durations >= p1) & (durations <= p99)]

        # Create violin plot
        parts = ax.violinplot([filtered], positions=[0], showmeans=True,
                              showmedians=True, widths=0.7)

        # Style the violin
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_edgecolor('#2c3e50')
            pc.set_alpha(0.7)

        parts['cmeans'].set_color('#e74c3c')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('#27ae60')
        parts['cmedians'].set_linewidth(2)

        # Add strip plot overlay (jittered points)
        jitter = np.random.normal(0, 0.08, len(filtered))
        ax.scatter(jitter, filtered, alpha=0.3, s=10, c='#2c3e50')

        # Add reference lines
        ax.axhline(y=150, color='#f39c12', linestyle='--', linewidth=1.5,
                   label='Rapid search threshold (150ms)')
        ax.axhline(y=400, color='#9b59b6', linestyle='--', linewidth=1.5,
                   label='Extended fixation threshold (400ms)')

        # Calculate and display statistics
        mean_dur = np.mean(filtered)
        median_dur = np.median(filtered)
        std_dur = np.std(filtered)

        stats_text = (f"Mean: {mean_dur:.0f}ms\n"
                      f"Median: {median_dur:.0f}ms\n"
                      f"Std: {std_dur:.0f}ms\n"
                      f"N: {len(durations)}")

        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Labels and styling
        ax.set_ylabel('Fixation Duration (ms)', fontsize=12)
        ax.set_title(f'Cognitive Fingerprint: Fixation Duration Distribution\n{subject_name}',
                     fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.legend(loc='upper left', fontsize=9)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_main_sequence(
        self,
        amplitudes: List[float],
        peak_velocities: List[float],
        subject_name: str,
        output_path: str,
        adaptive_threshold: Optional[float] = None,
        total_detected: Optional[int] = None,
        rejected_count: Optional[int] = None
    ) -> None:
        """
        Create saccade main sequence plot with power-law regression.

        The main sequence relationship (V = k * A^n) is a fundamental
        characteristic of saccadic eye movements. Deviations may indicate
        fatigue, medication effects, or neurological changes.

        Args:
            amplitudes: List of saccade amplitudes in degrees
            peak_velocities: List of peak velocities in deg/s
            subject_name: Subject identifier for title
            output_path: Output file path
            adaptive_threshold: MAD-based threshold used for detection (deg/s)
            total_detected: Total saccades detected before filtering
            rejected_count: Number of saccades rejected by physiological filtering
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        amplitudes = np.array(amplitudes)
        velocities = np.array(peak_velocities)

        # Filter invalid values
        valid = (amplitudes > 0) & (velocities > 0) & np.isfinite(amplitudes) & np.isfinite(velocities)
        amp = amplitudes[valid]
        vel = velocities[valid]

        if len(amp) < 10:
            ax.text(0.5, 0.5, 'Insufficient saccade data',
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return

        # Add physiological reference boundaries (subtle)
        ax.axvline(x=50, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(y=1000, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(48, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 50,
                '50° limit', fontsize=8, color='#7f8c8d', ha='right', va='bottom')
        ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else 40, 980,
                '1000 deg/s limit', fontsize=8, color='#7f8c8d', ha='right', va='top')

        # Scatter plot
        ax.scatter(amp, vel, alpha=0.4, s=30, c='#3498db', edgecolors='none',
                   label=f'Valid saccades (n={len(amp)})')

        # Power-law fit: V = k * A^n
        try:
            def power_law(x, k, n):
                return k * np.power(x, n)

            # Initial guess and bounds
            popt, pcov = curve_fit(power_law, amp, vel,
                                   p0=[100, 0.5], bounds=([0, 0], [2000, 2]),
                                   maxfev=5000)
            k, n = popt

            # Calculate R-squared
            predicted = power_law(amp, k, n)
            ss_res = np.sum((vel - predicted) ** 2)
            ss_tot = np.sum((vel - np.mean(vel)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Plot fit line
            amp_sorted = np.sort(amp)
            ax.plot(amp_sorted, power_law(amp_sorted, k, n),
                    'r-', linewidth=2, label=f'V = {k:.1f} × A^{n:.2f}')

            # Build statistics text
            stats_lines = [f'R² = {r_squared:.3f}', f'k = {k:.1f}, n = {n:.2f}']
            if adaptive_threshold is not None:
                stats_lines.append(f'Threshold: {adaptive_threshold:.1f} deg/s (MAD)')
            if total_detected is not None and rejected_count is not None:
                stats_lines.append(f'Filtered: {rejected_count}/{total_detected} rejected')

            ax.text(0.95, 0.05, '\n'.join(stats_lines),
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        except Exception as e:
            logger.warning(f"Could not fit power law: {e}")
            # Add linear regression as fallback
            slope, intercept, r, p, se = stats.linregress(amp, vel)
            x_fit = np.linspace(amp.min(), amp.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'r--', linewidth=2,
                    label=f'Linear fit (R² = {r**2:.3f})')

        # Labels and styling
        ax.set_xlabel('Saccade Amplitude (degrees)', fontsize=12)
        ax.set_ylabel('Peak Velocity (deg/s)', fontsize=12)
        ax.set_title(f'Main Sequence: Saccade Dynamics\n{subject_name}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_stress_timeline(
        self,
        pupil_timestamps: Optional[np.ndarray],
        pupil_residuals: Optional[np.ndarray],
        gyro_timestamps: Optional[np.ndarray],
        gyro_magnitude: Optional[np.ndarray],
        subject_name: str,
        output_path: str
    ) -> None:
        """
        Create stacked time-series plot of cognitive load indicators.

        Top panel: Pupil residuals (cognitive load proxy)
        Bottom panel: Head movement magnitude (motor stability)

        High values in both panels simultaneously may indicate stress/difficulty.

        Args:
            pupil_timestamps: Timestamps for pupil data (seconds)
            pupil_residuals: Luminance-adjusted pupil residuals (mm)
            gyro_timestamps: Timestamps for gyro data (seconds)
            gyro_magnitude: Total head rotation rate (deg/s)
            subject_name: Subject identifier for title
            output_path: Output file path
        """
        # Determine number of subplots needed
        has_pupil = pupil_timestamps is not None and pupil_residuals is not None
        has_gyro = gyro_timestamps is not None and gyro_magnitude is not None
        n_plots = int(has_pupil) + int(has_gyro)

        if n_plots == 0:
            logger.warning("No data available for stress timeline")
            return

        fig, axes = plt.subplots(n_plots, 1, figsize=self.figure_size, sharex=True)
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Pupil residuals panel
        if has_pupil and len(pupil_timestamps) > 0:
            ax = axes[plot_idx]
            plot_idx += 1

            # Normalize timestamps to start at 0
            t = pupil_timestamps - pupil_timestamps[0]

            # Calculate rolling mean
            window_samples = int(self.rolling_window_s * len(t) / t[-1]) if t[-1] > 0 else 50
            window_samples = max(10, min(window_samples, len(t) // 4))

            # Pad for rolling mean calculation
            padded = np.pad(pupil_residuals, (window_samples // 2, window_samples // 2), mode='edge')
            rolling_mean = np.convolve(padded, np.ones(window_samples) / window_samples, mode='valid')

            # Ensure same length (trim if needed)
            if len(rolling_mean) > len(t):
                rolling_mean = rolling_mean[:len(t)]
            elif len(rolling_mean) < len(t):
                rolling_mean = np.pad(rolling_mean, (0, len(t) - len(rolling_mean)), mode='edge')

            # Plot raw and smoothed
            ax.fill_between(t, pupil_residuals, alpha=0.3, color='#3498db', label='Raw')
            ax.plot(t, rolling_mean, color='#2c3e50', linewidth=1.5,
                    label=f'{self.rolling_window_s}s rolling mean')

            # Highlight high-load regions (>1 std above mean)
            threshold = np.mean(pupil_residuals) + np.std(pupil_residuals)
            high_load = rolling_mean > threshold
            ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=high_load, alpha=0.2, color='#e74c3c',
                            label='High cognitive load')

            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_ylabel('Pupil Residual (mm)', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title('Cognitive Load (Pupil)', fontsize=11)

        # Gyro magnitude panel
        if has_gyro and len(gyro_timestamps) > 0:
            ax = axes[plot_idx]

            # Normalize timestamps to start at 0
            t = gyro_timestamps - gyro_timestamps[0]

            # Calculate rolling mean
            window_samples = int(self.rolling_window_s * len(t) / t[-1]) if t[-1] > 0 else 50
            window_samples = max(10, min(window_samples, len(t) // 4))

            padded = np.pad(gyro_magnitude, (window_samples // 2, window_samples // 2), mode='edge')
            rolling_mean = np.convolve(padded, np.ones(window_samples) / window_samples, mode='valid')

            if len(rolling_mean) > len(t):
                rolling_mean = rolling_mean[:len(t)]
            elif len(rolling_mean) < len(t):
                rolling_mean = np.pad(rolling_mean, (0, len(t) - len(rolling_mean)), mode='edge')

            # Plot
            ax.fill_between(t, gyro_magnitude, alpha=0.3, color='#27ae60', label='Raw')
            ax.plot(t, rolling_mean, color='#2c3e50', linewidth=1.5,
                    label=f'{self.rolling_window_s}s rolling mean')

            # Highlight high-motion regions
            threshold = np.mean(gyro_magnitude) + np.std(gyro_magnitude)
            high_motion = rolling_mean > threshold
            ax.fill_between(t, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else np.max(gyro_magnitude),
                            where=high_motion, alpha=0.2, color='#f39c12',
                            label='High head motion')

            ax.set_ylabel('Head Motion (deg/s)', fontsize=11)
            ax.set_xlabel('Time (seconds)', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title('Motor Stability (Head)', fontsize=11)
            ax.set_ylim(bottom=0)

        # Overall title
        fig.suptitle(f'Stress Timeline: Cognitive Load & Motor Stability\n{subject_name}',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_stability_radar(
        self,
        gyro_x: np.ndarray,
        gyro_y: np.ndarray,
        subject_name: str,
        output_path: str
    ) -> None:
        """
        Create polar histogram of head movement directions.

        Shows the distribution of head movement directions:
        - Up/Down: Pitch movements (gyro_x)
        - Left/Right: Yaw movements (gyro_y)

        A uniform distribution suggests random head movements.
        Clustered patterns may indicate consistent visual search strategies.

        Args:
            gyro_x: Pitch (up/down) angular velocity in deg/s
            gyro_y: Yaw (left/right) angular velocity in deg/s
            subject_name: Subject identifier for title
            output_path: Output file path
        """
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Calculate direction angles from gyro_x (pitch) and gyro_y (yaw)
        angles = np.arctan2(gyro_y, gyro_x)

        # Calculate magnitudes for weighting
        magnitudes = np.sqrt(gyro_x**2 + gyro_y**2)

        # Filter out very small movements (noise)
        threshold = np.percentile(magnitudes, 25)
        significant = magnitudes > threshold
        angles = angles[significant]
        magnitudes = magnitudes[significant]

        if len(angles) < 10:
            ax.text(0, 0, 'Insufficient data', ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return

        # Create 36-bin polar histogram (10-degree bins)
        n_bins = 36
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)

        # Magnitude-weighted histogram
        hist, _ = np.histogram(angles, bins=bin_edges, weights=magnitudes)
        hist = hist / hist.sum()  # Normalize

        # Plot bars
        width = 2 * np.pi / n_bins
        bars = ax.bar(bin_edges[:-1] + width/2, hist, width=width, bottom=0,
                      alpha=0.7, color='#3498db', edgecolor='#2c3e50')

        # Add direction labels
        ax.set_theta_zero_location('N')  # 0 degrees at top
        ax.set_theta_direction(-1)  # Clockwise

        # Custom direction labels
        direction_labels = {
            0: 'Up',
            np.pi/2: 'Right',
            np.pi: 'Down',
            -np.pi/2: 'Left'
        }

        ax.set_xticks([0, np.pi/2, np.pi, -np.pi/2])
        ax.set_xticklabels(['Up (Pitch+)', 'Right (Yaw+)', 'Down (Pitch-)', 'Left (Yaw-)'],
                          fontsize=10)

        # Add statistics
        # Calculate dominant direction
        dominant_bin = np.argmax(hist)
        dominant_angle = bin_edges[dominant_bin] + width/2
        dominant_pct = hist[dominant_bin] * 100

        # Calculate uniformity (0=uniform, 1=concentrated)
        uniformity = 1 - (np.std(hist) / (1/n_bins))  # Normalized

        stats_text = (f"Total movements: {len(angles)}\n"
                      f"Dominant direction: {np.degrees(dominant_angle):.0f}°\n"
                      f"Concentration: {(1-uniformity)*100:.1f}%")

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f'Stability Radar: Head Movement Distribution\n{subject_name}',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_kernel_fit_plot(
        self,
        kernel_model,
        pupil: np.ndarray,
        luminance: np.ndarray,
        timestamps: np.ndarray,
        convolved_luminance: np.ndarray,
        metrics: dict,
        subject_name: str,
        output_path: str
    ) -> None:
        """
        Create visualization of the per-subject fitted PLR kernel.

        Shows four panels:
        1. Top-left: Impulse Response Function (kernel shape)
        2. Top-right: Time series overlay (luminance, convolved, pupil)
        3. Bottom-left: Instantaneous regression scatter
        4. Bottom-right: Convolved regression scatter

        Args:
            kernel_model: PupilLuminanceKernel instance with fitted params
            pupil: Pupil diameter time series
            luminance: Frame luminance time series
            timestamps: Timestamps in seconds
            convolved_luminance: Pre-computed convolved luminance
            metrics: Dictionary with regression metrics
            subject_name: Subject identifier for title
            output_path: Output file path
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Normalize timestamps
        t = timestamps - timestamps[0]

        # Get kernel parameters
        t_max = metrics.get('kernel_t_max_ms', 512)
        n = metrics.get('kernel_n', 10.1)
        is_fitted = metrics.get('kernel_is_fitted', False)
        r2_inst = metrics.get('regression_r_squared', 0)
        r2_conv = metrics.get('regression_r_squared_convolved', 0)

        # ===== Panel 1: Impulse Response Function =====
        ax1 = axes[0, 0]
        try:
            kernel_time, kernel_vals = kernel_model.get_kernel_for_plotting()
            ax1.fill_between(kernel_time, kernel_vals, alpha=0.3, color='#3498db')
            ax1.plot(kernel_time, kernel_vals, color='#2c3e50', linewidth=2)

            # Mark peak
            peak_idx = np.argmax(kernel_vals)
            ax1.axvline(x=kernel_time[peak_idx], color='#e74c3c', linestyle='--',
                       linewidth=1.5, label=f't_max = {t_max:.0f} ms')

            # Add canonical reference
            canonical_kernel = kernel_model.create_erlang_kernel(512, 10.1)
            canonical_time = np.arange(len(canonical_kernel)) * (1000.0 / kernel_model.sampling_rate)
            ax1.plot(canonical_time, canonical_kernel, color='#95a5a6', linestyle=':',
                    linewidth=1.5, alpha=0.7, label='Canonical (512ms, 10.1)')

        except Exception as e:
            ax1.text(0.5, 0.5, f'Kernel plot error: {e}',
                    transform=ax1.transAxes, ha='center', va='center')

        ax1.set_xlabel('Time (ms)', fontsize=11)
        ax1.set_ylabel('Response Weight', fontsize=11)
        ax1.set_title(f'PLR Impulse Response Function\nt_max={t_max:.0f}ms, n={n:.1f} ({"fitted" if is_fitted else "canonical"})',
                     fontsize=11, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_xlim(0, 2000)

        # ===== Panel 2: Time Series Overlay =====
        ax2 = axes[0, 1]

        # Subsample for plotting if too many points (avoid slow rendering)
        max_points = 5000
        if len(t) > max_points:
            step = len(t) // max_points
            t_plot = t[::step]
            lum_plot = luminance[::step]
            conv_plot = convolved_luminance[::step]
            pupil_plot = pupil[::step]
        else:
            t_plot, lum_plot, conv_plot, pupil_plot = t, luminance, convolved_luminance, pupil

        # Normalize for overlay (z-score each signal)
        lum_norm = (lum_plot - np.nanmean(lum_plot)) / np.nanstd(lum_plot)
        conv_norm = (conv_plot - np.nanmean(conv_plot)) / np.nanstd(conv_plot)
        pupil_norm = (pupil_plot - np.nanmean(pupil_plot)) / np.nanstd(pupil_plot)

        ax2.plot(t_plot, lum_norm, color='#f39c12', alpha=0.5, linewidth=0.8,
                label='Luminance (z-scored)')
        ax2.plot(t_plot, conv_norm, color='#27ae60', linewidth=1.5,
                label='Convolved Luminance')
        ax2.plot(t_plot, pupil_norm, color='#3498db', alpha=0.7, linewidth=0.8,
                label='Pupil (z-scored)')

        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Z-scored Signal', fontsize=11)
        ax2.set_title('Temporal Alignment: Luminance vs Pupil', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)

        # Show only first 60 seconds for clarity
        ax2.set_xlim(0, min(60, t[-1]))

        # ===== Panel 3: Instantaneous Regression =====
        ax3 = axes[1, 0]

        # Subsample for scatter
        if len(luminance) > 2000:
            idx = np.random.choice(len(luminance), 2000, replace=False)
        else:
            idx = np.arange(len(luminance))

        valid = ~(np.isnan(luminance[idx]) | np.isnan(pupil[idx]))
        lum_scatter = luminance[idx][valid]
        pupil_scatter = pupil[idx][valid]

        ax3.scatter(lum_scatter, pupil_scatter, alpha=0.3, s=10, c='#3498db', edgecolors='none')

        # Regression line
        slope_inst = metrics.get('regression_slope', 0)
        intercept_inst = metrics.get('regression_intercept', 0)
        x_line = np.array([np.nanmin(luminance), np.nanmax(luminance)])
        ax3.plot(x_line, slope_inst * x_line + intercept_inst, 'r-', linewidth=2,
                label=f'R² = {r2_inst:.4f}')

        ax3.set_xlabel('Frame Luminance', fontsize=11)
        ax3.set_ylabel('Pupil Diameter (mm)', fontsize=11)
        ax3.set_title('Instantaneous Regression (Baseline)', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)

        # ===== Panel 4: Convolved Regression =====
        ax4 = axes[1, 1]

        valid_conv = ~(np.isnan(convolved_luminance[idx]) | np.isnan(pupil[idx]))
        conv_scatter = convolved_luminance[idx][valid_conv]
        pupil_scatter_conv = pupil[idx][valid_conv]

        ax4.scatter(conv_scatter, pupil_scatter_conv, alpha=0.3, s=10, c='#27ae60', edgecolors='none')

        # Regression line
        slope_conv = metrics.get('regression_slope_convolved', 0)
        intercept_conv = metrics.get('regression_intercept_convolved', 0)
        x_line_conv = np.array([np.nanmin(convolved_luminance), np.nanmax(convolved_luminance)])
        ax4.plot(x_line_conv, slope_conv * x_line_conv + intercept_conv, 'r-', linewidth=2,
                label=f'R² = {r2_conv:.4f}')

        ax4.set_xlabel('Convolved Luminance', fontsize=11)
        ax4.set_ylabel('Pupil Diameter (mm)', fontsize=11)
        ax4.set_title('Convolved Regression (Temporal Kernel)', fontsize=11, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=10)

        # ===== Overall title and improvement annotation =====
        r2_improvement = metrics.get('r_squared_improvement', 0)
        improvement_pct = metrics.get('r_squared_improvement_pct', 0)

        fig.suptitle(
            f'PLR Temporal Kernel Analysis: {subject_name}\n'
            f'R² Improvement: {r2_inst:.4f} → {r2_conv:.4f} (+{r2_improvement:.4f}, {improvement_pct:+.0f}%)',
            fontsize=14, fontweight='bold', y=1.02
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()


def create_visualizations_for_analyzer(
    analyzer,
    output_dir: str,
    subject_name: str,
    config: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to create all visualizations from an analyzer.

    Args:
        analyzer: WholeSessionAnalyzer instance with computed metrics
        output_dir: Directory to save visualizations
        subject_name: Subject identifier
        config: Optional visualizer configuration

    Returns:
        Dictionary with created files and any errors
    """
    visualizer = GazeVisualizer(config)
    return visualizer.create_all_visualizations(analyzer, output_dir, subject_name)
