"""
Cohort Visualization Module for Cross-Subject Comparisons

Creates visualizations comparing metrics across all subjects in the cohort.
"""

from scipy import stats
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')

from ..logging_config import get_logger

logger = get_logger(__name__)


class CohortVisualizer:
    """
    Creates cross-subject comparison visualizations.

    Generates:
    - Metric distribution box plots
    - Subject ranking charts
    - Correlation heatmap
    - Cohort summary dashboard
    - Main sequence overlay plot
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the cohort visualizer.

        Args:
            config: Configuration dictionary with optional keys:
                - figure_size: Tuple (width, height) in inches
                - dpi: Output resolution
        """
        config = config or {}
        self.figure_size = tuple(config.get('figure_size', (14, 10)))
        self.dpi = config.get('dpi', 300)

        plt.style.use('seaborn-v0_8-whitegrid')

        # Color palette for subjects
        self.subject_colors = plt.cm.tab20.colors

    def create_all_visualizations(
        self,
        df: pd.DataFrame,
        output_dir: str,
        logs_dir: str = None
    ) -> Dict[str, str]:
        """
        Create all cohort visualizations.

        Args:
            df: DataFrame with subjects as rows, metrics as columns
            output_dir: Directory to save visualizations
            logs_dir: Directory with analysis JSON files (for main sequence overlay)

        Returns:
            Dictionary mapping visualization name to output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created = {}

        # 1. Metric Distribution Box Plots
        try:
            path = output_dir / "cohort_metric_distributions.png"
            self.create_metric_comparison_boxplots(df, str(path))
            created['metric_distributions'] = str(path)
            logger.info(f"  Created: {path.name}")
        except Exception as e:
            logger.warning(f"  Failed metric distributions: {e}")

        # 2. Correlation Heatmap
        try:
            path = output_dir / "cohort_correlation_matrix.png"
            self.create_correlation_heatmap(df, str(path))
            created['correlation_matrix'] = str(path)
            logger.info(f"  Created: {path.name}")
        except Exception as e:
            logger.warning(f"  Failed correlation heatmap: {e}")

        # 3. Subject Rankings
        ranking_metrics = [
            ('fixation_rate_hz', 'Fixation Rate'),
            ('std_residual', 'Cognitive Load (Pupil Std)'),
            ('mean_rotation_rate', 'Motor Stability (Head Motion)'),
        ]

        for metric, label in ranking_metrics:
            if metric in df.columns:
                try:
                    safe_name = metric.replace('_', '-')
                    path = output_dir / f"cohort_ranking_{safe_name}.png"
                    self.create_subject_ranking_chart(df, metric, label, str(path))
                    created[f'ranking_{metric}'] = str(path)
                    logger.info(f"  Created: {path.name}")
                except Exception as e:
                    logger.warning(f"  Failed ranking {metric}: {e}")

        # 4. Cohort Dashboard
        try:
            path = output_dir / "cohort_summary_dashboard.png"
            self.create_cohort_dashboard(df, str(path))
            created['summary_dashboard'] = str(path)
            logger.info(f"  Created: {path.name}")
        except Exception as e:
            logger.warning(f"  Failed cohort dashboard: {e}")

        # 5. Main Sequence Overlay (if logs_dir provided)
        if logs_dir:
            try:
                path = output_dir / "cohort_main_sequence_overlay.png"
                self.create_main_sequence_overlay(logs_dir, df, str(path))
                created['main_sequence_overlay'] = str(path)
                logger.info(f"  Created: {path.name}")
            except Exception as e:
                logger.warning(f"  Failed main sequence overlay: {e}")

        return created

    def create_metric_comparison_boxplots(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        Create box plots comparing key metrics across all subjects.

        Args:
            df: DataFrame with metrics
            output_path: Output file path
        """
        # Select key metrics for visualization
        metrics_to_plot = [
            ('fixation_rate_hz', 'Fixation Rate\n(Hz)'),
            ('mean_fixation_duration_ms', 'Fixation Duration\n(ms)'),
            ('saccade_count', 'Saccade Count'),
            ('adaptive_threshold_deg_s', 'Adaptive Threshold\n(deg/s)'),
            ('saccade_amplitude_mean_deg', 'Saccade Amplitude\n(deg)'),
            ('main_sequence_r_squared', 'Main Sequence\nR²'),
            ('std_residual', 'Pupil Residual Std\n(mm)'),
            ('raw_pupil_mean', 'Pupil Size\n(mm)'),
            ('total_rotation_deg', 'Total Head Rotation\n(deg)'),
            ('mean_rotation_rate', 'Head Rotation Rate\n(deg/s)'),
            ('luminance_mean', 'Frame Luminance'),
            ('fixation_proportion', 'Fixation Proportion'),
        ]

        # Filter to available metrics
        available = [(m, l) for m, l in metrics_to_plot if m in df.columns]

        if not available:
            logger.warning("No metrics available for box plots")
            return

        n_metrics = len(available)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_metrics == 1 else axes.flatten()

        for i, (metric, label) in enumerate(available):
            ax = axes[i]
            values = df[metric].dropna()

            if len(values) > 0:
                # Box plot with individual points
                bp = ax.boxplot([values], widths=0.6, patch_artist=True)
                bp['boxes'][0].set_facecolor('#3498db')
                bp['boxes'][0].set_alpha(0.7)

                # Overlay individual points with jitter
                jitter = np.random.normal(1, 0.04, len(values))
                ax.scatter(jitter, values, alpha=0.6, s=40, c='#2c3e50', zorder=3)

                # Add mean marker
                ax.scatter([1], [values.mean()], marker='D', s=80, c='#e74c3c',
                          zorder=4, label=f'Mean: {values.mean():.2f}')

            ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([])
            ax.set_title(f'n={len(values)}', fontsize=9, style='italic')

        # Hide unused axes
        for i in range(len(available), len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f'Cohort Metric Distributions (N={len(df)} subjects)',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_subject_ranking_chart(
        self,
        df: pd.DataFrame,
        metric: str,
        metric_label: str,
        output_path: str
    ) -> None:
        """
        Create horizontal bar chart ranking subjects by a metric.

        Args:
            df: DataFrame with metrics
            metric: Column name to rank by
            metric_label: Human-readable metric name
            output_path: Output file path
        """
        if metric not in df.columns:
            return

        values = df[metric].dropna().sort_values()

        fig, ax = plt.subplots(figsize=(10, max(6, len(values) * 0.4)))

        # Color bars by value (gradient)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))

        bars = ax.barh(range(len(values)), values.values, color=colors, edgecolor='#2c3e50')

        # Add value labels
        for i, (idx, val) in enumerate(values.items()):
            ax.text(val + values.max() * 0.02, i, f'{val:.2f}',
                   va='center', fontsize=9)

        # Format subject IDs (shorter labels)
        short_labels = [s[:8] + '...' if len(s) > 12 else s for s in values.index]
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(short_labels, fontsize=9)

        ax.set_xlabel(metric_label, fontsize=11)
        ax.set_title(f'Subject Ranking: {metric_label}\n(N={len(values)} subjects)',
                    fontsize=12, fontweight='bold')

        # Add mean line
        mean_val = values.mean()
        ax.axvline(x=mean_val, color='#e74c3c', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.2f}')
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        Create heatmap showing correlations between metrics.

        Args:
            df: DataFrame with metrics
            output_path: Output file path
        """
        # Select numeric columns with enough valid data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        valid_cols = [c for c in numeric_cols if df[c].notna().sum() >= 3]

        if len(valid_cols) < 2:
            logger.warning("Not enough valid columns for correlation heatmap")
            return

        # Compute correlation matrix
        corr_matrix = df[valid_cols].corr()

        # Create shorter labels
        short_labels = {
            'fixation_rate_hz': 'Fix Rate',
            'mean_fixation_duration_ms': 'Fix Duration',
            'saccade_count': 'Saccade N',
            'adaptive_threshold_deg_s': 'Threshold',
            'saccade_amplitude_mean_deg': 'Sacc Amp',
            'main_sequence_r_squared': 'Main Seq R²',
            'std_residual': 'Pupil Std',
            'raw_pupil_mean': 'Pupil Mean',
            'total_rotation_deg': 'Head Rot',
            'mean_rotation_rate': 'Rot Rate',
            'fixation_proportion': 'Fix Prop',
            'saccade_proportion': 'Sacc Prop',
            'luminance_mean': 'Luminance',
            'regression_r_squared': 'Lum-Pupil R²',
        }

        labels = [short_labels.get(c, c[:10]) for c in corr_matrix.columns]

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5,
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={'size': 8}, ax=ax)

        ax.set_title(f'Metric Correlation Matrix (N={len(df)} subjects)',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_cohort_dashboard(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        Create multi-panel cohort summary dashboard.

        Args:
            df: DataFrame with metrics
            output_path: Output file path
        """
        fig = plt.figure(figsize=(16, 12))

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Panel 1: Recording duration distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'total_duration_s' in df.columns:
            durations = df['total_duration_s'].dropna() / 60  # Convert to minutes
            ax1.hist(durations, bins=10, color='#3498db', edgecolor='#2c3e50', alpha=0.7)
            ax1.axvline(durations.mean(), color='#e74c3c', linestyle='--',
                       label=f'Mean: {durations.mean():.1f} min')
            ax1.set_xlabel('Duration (minutes)')
            ax1.set_ylabel('Count')
            ax1.set_title('Recording Durations')
            ax1.legend()

        # Panel 2: Fixation rate distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if 'fixation_rate_hz' in df.columns:
            vals = df['fixation_rate_hz'].dropna()
            ax2.hist(vals, bins=10, color='#27ae60', edgecolor='#2c3e50', alpha=0.7)
            ax2.axvline(vals.mean(), color='#e74c3c', linestyle='--',
                       label=f'Mean: {vals.mean():.2f} Hz')
            ax2.set_xlabel('Fixation Rate (Hz)')
            ax2.set_ylabel('Count')
            ax2.set_title('Oculometric Efficiency')
            ax2.legend()

        # Panel 3: Main sequence R² distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if 'main_sequence_r_squared' in df.columns:
            vals = df['main_sequence_r_squared'].dropna()
            ax3.hist(vals, bins=10, color='#9b59b6', edgecolor='#2c3e50', alpha=0.7)
            ax3.axvline(vals.mean(), color='#e74c3c', linestyle='--',
                       label=f'Mean: {vals.mean():.3f}')
            ax3.set_xlabel('Main Sequence R²')
            ax3.set_ylabel('Count')
            ax3.set_title('Saccade Detection Quality')
            ax3.legend()

        # Panel 4: Cognitive load (pupil std)
        ax4 = fig.add_subplot(gs[1, 0])
        if 'std_residual' in df.columns:
            vals = df['std_residual'].dropna()
            ax4.hist(vals, bins=10, color='#e67e22', edgecolor='#2c3e50', alpha=0.7)
            ax4.axvline(vals.mean(), color='#e74c3c', linestyle='--',
                       label=f'Mean: {vals.mean():.4f} mm')
            ax4.set_xlabel('Pupil Residual Std (mm)')
            ax4.set_ylabel('Count')
            ax4.set_title('Cognitive Load Proxy')
            ax4.legend()

        # Panel 5: Motor stability (head rotation)
        ax5 = fig.add_subplot(gs[1, 1])
        if 'mean_rotation_rate' in df.columns:
            vals = df['mean_rotation_rate'].dropna()
            ax5.hist(vals, bins=10, color='#1abc9c', edgecolor='#2c3e50', alpha=0.7)
            ax5.axvline(vals.mean(), color='#e74c3c', linestyle='--',
                       label=f'Mean: {vals.mean():.1f} deg/s')
            ax5.set_xlabel('Mean Rotation Rate (deg/s)')
            ax5.set_ylabel('Count')
            ax5.set_title('Motor Stability')
            ax5.legend()

        # Panel 6: Saccade count distribution
        ax6 = fig.add_subplot(gs[1, 2])
        if 'saccade_count' in df.columns:
            vals = df['saccade_count'].dropna()
            ax6.hist(vals, bins=10, color='#34495e', edgecolor='#2c3e50', alpha=0.7)
            ax6.axvline(vals.mean(), color='#e74c3c', linestyle='--',
                       label=f'Mean: {vals.mean():.0f}')
            ax6.set_xlabel('Valid Saccade Count')
            ax6.set_ylabel('Count')
            ax6.set_title('Saccade Detection')
            ax6.legend()

        # Panel 7-9: Key metric scatter plots
        # Fixation rate vs Cognitive load
        ax7 = fig.add_subplot(gs[2, 0])
        if 'fixation_rate_hz' in df.columns and 'std_residual' in df.columns:
            x = df['fixation_rate_hz'].dropna()
            y = df['std_residual'].reindex(x.index).dropna()
            x = x.reindex(y.index)
            if len(x) > 2:
                ax7.scatter(x, y, c='#3498db', s=60, alpha=0.7, edgecolors='#2c3e50')
                # Add correlation
                r, p = stats.pearsonr(x, y)
                ax7.set_title(f'r = {r:.3f}, p = {p:.3f}', fontsize=10)
            ax7.set_xlabel('Fixation Rate (Hz)')
            ax7.set_ylabel('Pupil Residual Std (mm)')

        # Main sequence R² vs Saccade count
        ax8 = fig.add_subplot(gs[2, 1])
        if 'main_sequence_r_squared' in df.columns and 'saccade_count' in df.columns:
            x = df['saccade_count'].dropna()
            y = df['main_sequence_r_squared'].reindex(x.index).dropna()
            x = x.reindex(y.index)
            if len(x) > 2:
                ax8.scatter(x, y, c='#9b59b6', s=60, alpha=0.7, edgecolors='#2c3e50')
                r, p = stats.pearsonr(x, y)
                ax8.set_title(f'r = {r:.3f}, p = {p:.3f}', fontsize=10)
            ax8.set_xlabel('Saccade Count')
            ax8.set_ylabel('Main Sequence R²')

        # Summary statistics text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = f"COHORT SUMMARY\n"
        summary_text += f"{'='*30}\n\n"
        summary_text += f"Subjects: {len(df)}\n\n"

        if 'total_duration_s' in df.columns:
            total_hrs = df['total_duration_s'].sum() / 3600
            summary_text += f"Total Recording: {total_hrs:.1f} hours\n\n"

        if 'main_sequence_r_squared' in df.columns:
            r2_mean = df['main_sequence_r_squared'].mean()
            summary_text += f"Mean Main Seq R²: {r2_mean:.3f}\n"

        if 'saccade_count' in df.columns:
            total_saccades = df['saccade_count'].sum()
            summary_text += f"Total Saccades: {total_saccades:,.0f}\n"

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

        fig.suptitle(f'Cohort Analysis Dashboard (N={len(df)} subjects)',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def create_main_sequence_overlay(
        self,
        logs_dir: str,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        Overlay all subjects' saccade data on a single main sequence plot.

        Args:
            logs_dir: Directory with whole_session_analysis.json files
            df: DataFrame with subject metrics (for R² values)
            output_path: Output file path
        """
        logs_dir = Path(logs_dir)
        processed_dir = logs_dir.parent.parent / 'data' / 'processed'

        fig, ax = plt.subplots(figsize=(12, 10))

        all_amplitudes = []
        all_velocities = []
        subject_data = []

        # Load saccade data from each subject's CSV
        for i, subject_id in enumerate(sorted(df.index)):
            csv_path = processed_dir / subject_id / f"{subject_id}_final_gaze_data.csv"

            if not csv_path.exists():
                continue

            try:
                # Re-run saccade detection to get raw data
                from .adaptive_saccade_detector import AdaptiveSaccadeDetector

                subject_df = pd.read_csv(csv_path)
                detector = AdaptiveSaccadeDetector()
                saccades = detector.detect_saccades(subject_df)

                valid_saccades = [s for s in saccades if s.is_valid]

                if not valid_saccades:
                    continue

                amps = [s.amplitude_deg for s in valid_saccades]
                vels = [s.peak_velocity_deg_s for s in valid_saccades]

                all_amplitudes.extend(amps)
                all_velocities.extend(vels)

                # Plot this subject's data
                color = self.subject_colors[i % len(self.subject_colors)]
                ax.scatter(amps, vels, alpha=0.3, s=15, c=[color],
                          label=f'{subject_id[:8]}... (n={len(amps)})')

                subject_data.append({
                    'subject': subject_id,
                    'n_saccades': len(amps),
                    'color': color
                })

            except Exception as e:
                logger.debug(f"Could not load saccade data for {subject_id}: {e}")

        if not all_amplitudes:
            ax.text(0.5, 0.5, 'No saccade data available',
                   transform=ax.transAxes, ha='center', va='center')
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return

        # Fit population-level power law
        all_amp = np.array(all_amplitudes)
        all_vel = np.array(all_velocities)

        valid = (all_amp > 0) & (all_vel > 0)
        amp_valid = all_amp[valid]
        vel_valid = all_vel[valid]

        try:
            def power_law(x, k, n):
                return k * np.power(x, n)

            popt, _ = curve_fit(power_law, amp_valid, vel_valid,
                               p0=[100, 0.5], bounds=([0, 0], [2000, 2]),
                               maxfev=10000)
            k, n = popt

            # Calculate R²
            predicted = power_law(amp_valid, k, n)
            ss_res = np.sum((vel_valid - predicted) ** 2)
            ss_tot = np.sum((vel_valid - np.mean(vel_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Plot fit line
            amp_sorted = np.linspace(0.1, amp_valid.max(), 100)
            ax.plot(amp_sorted, power_law(amp_sorted, k, n),
                   'k-', linewidth=3, label=f'Population fit: V = {k:.1f} × A^{n:.2f}')

            # Add stats box
            stats_text = (f'Population Statistics\n'
                         f'N subjects: {len(subject_data)}\n'
                         f'N saccades: {len(amp_valid):,}\n'
                         f'R² = {r_squared:.3f}\n'
                         f'k = {k:.1f}, n = {n:.2f}')

            ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        except Exception as e:
            logger.warning(f"Could not fit population power law: {e}")

        # Add physiological limits
        ax.axvline(x=50, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(y=1000, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.7)

        ax.set_xlabel('Saccade Amplitude (degrees)', fontsize=12)
        ax.set_ylabel('Peak Velocity (deg/s)', fontsize=12)
        ax.set_title(f'Population Main Sequence: All Subjects Combined\n'
                    f'({len(subject_data)} subjects, {len(amp_valid):,} saccades)',
                    fontsize=14, fontweight='bold')

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # Legend (limit to avoid overcrowding)
        if len(subject_data) <= 10:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()


def create_cohort_visualizations(
    df: pd.DataFrame,
    output_dir: str,
    logs_dir: str = None,
    config: Dict = None
) -> Dict[str, str]:
    """
    Convenience function to create all cohort visualizations.

    Args:
        df: DataFrame with subject metrics
        output_dir: Directory for output files
        logs_dir: Directory with analysis JSON files
        config: Visualizer configuration

    Returns:
        Dictionary of created file paths
    """
    visualizer = CohortVisualizer(config)
    return visualizer.create_all_visualizations(df, output_dir, logs_dir)
