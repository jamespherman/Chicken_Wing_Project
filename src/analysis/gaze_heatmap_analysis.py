#!/usr/bin/env python3
"""
gaze_heatmap_analysis.py - Generate 2D gaze heatmaps for visual analysis

This script creates heatmaps from processed gaze data to provide
immediate visual feedback on gaze patterns and data quality.

Features:
- Multiple visualization styles (heatmap, scatter, contour)
- Customizable color schemes and resolution
- Statistics overlay and data quality metrics
- Batch processing integration
- Individual and comparative analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import json
import traceback
from scipy import stats
from scipy.ndimage import gaussian_filter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("viridis")


class GazeHeatmapAnalyzer:
    """
    Analyzer class for creating gaze heatmaps from processed CSV data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the analyzer with configuration options.
        
        Args:
            config (dict): Configuration dictionary for visualization parameters
        """
        # Default configuration
        self.config = {
            # Visualization settings
            'figure_size': (12, 8),
            'dpi': 300,
            'color_scheme': 'viridis',  # viridis, plasma, inferno, magma, hot, etc.
            'heatmap_bins': 50,  # Resolution of the heatmap
            'gaussian_sigma': 1.0,  # Smoothing for heatmap
            
            # Output settings
            'output_format': 'png',
            'save_stats': True,
            'show_title': True,
            'show_stats_overlay': True,
            'show_colorbar': True,
            
            # Data filtering
            'workspace_bounds': None,  # (min_x, max_x, min_y, max_y) or None for auto
            'outlier_percentile': 99,  # Remove extreme outliers beyond this percentile
            'min_valid_points': 100,  # Minimum valid points required for visualization
            
            # Multiple visualization types
            'create_heatmap': True,
            'create_scatter': True,
            'create_contour': True,
            'create_combined': True
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
    
    def load_gaze_data(self, csv_path):
        """
        Load and validate gaze data from CSV file.
        
        Args:
            csv_path (str): Path to the final gaze CSV file
            
        Returns:
            Loaded gaze data with validation
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['gaze_timestamp', 'transformed_gaze_x', 'transformed_gaze_y']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            print(f"Loaded {len(df)} gaze samples from {Path(csv_path).name}")
            return df
            
        except Exception as e:
            print(f"Error loading gaze data from {csv_path}: {e}")
            return None
    
    def filter_valid_gaze_data(self, df):
        """
        Filter and clean gaze data for visualization.
        
        Args:
            df: Raw gaze data
            
        Returns:
            tuple: (filtered_df, stats_dict)
        """
        original_count = len(df)
        
        # Remove NaN values
        valid_df = df.dropna(subset=['transformed_gaze_x', 'transformed_gaze_y'])
        valid_count = len(valid_df)
        
        if valid_count == 0:
            return None, {'error': 'No valid gaze points found'}
        
        # Remove outliers if specified
        if self.config['outlier_percentile'] < 100:
            percentile = self.config['outlier_percentile']
            
            x_lower = np.percentile(valid_df['transformed_gaze_x'], (100 - percentile) / 2)
            x_upper = np.percentile(valid_df['transformed_gaze_x'], 100 - (100 - percentile) / 2)
            y_lower = np.percentile(valid_df['transformed_gaze_y'], (100 - percentile) / 2)
            y_upper = np.percentile(valid_df['transformed_gaze_y'], 100 - (100 - percentile) / 2)
            
            valid_df = valid_df[
                (valid_df['transformed_gaze_x'] >= x_lower) &
                (valid_df['transformed_gaze_x'] <= x_upper) &
                (valid_df['transformed_gaze_y'] >= y_lower) &
                (valid_df['transformed_gaze_y'] <= y_upper)
            ]
        
        # Apply workspace bounds if specified
        if self.config['workspace_bounds']:
            min_x, max_x, min_y, max_y = self.config['workspace_bounds']
            valid_df = valid_df[
                (valid_df['transformed_gaze_x'] >= min_x) &
                (valid_df['transformed_gaze_x'] <= max_x) &
                (valid_df['transformed_gaze_y'] >= min_y) &
                (valid_df['transformed_gaze_y'] <= max_y)
            ]
        
        filtered_count = len(valid_df)
        
        # Calculate statistics
        stats = {
            'original_samples': original_count,
            'valid_samples': valid_count,
            'filtered_samples': filtered_count,
            'valid_percentage': (valid_count / original_count * 100) if original_count > 0 else 0,
            'filtered_percentage': (filtered_count / original_count * 100) if original_count > 0 else 0,
            'x_range': (valid_df['transformed_gaze_x'].min(), valid_df['transformed_gaze_x'].max()) if filtered_count > 0 else (0, 0),
            'y_range': (valid_df['transformed_gaze_y'].min(), valid_df['transformed_gaze_y'].max()) if filtered_count > 0 else (0, 0),
            'x_mean': valid_df['transformed_gaze_x'].mean() if filtered_count > 0 else 0,
            'y_mean': valid_df['transformed_gaze_y'].mean() if filtered_count > 0 else 0,
            'x_std': valid_df['transformed_gaze_x'].std() if filtered_count > 0 else 0,
            'y_std': valid_df['transformed_gaze_y'].std() if filtered_count > 0 else 0
        }
        
        print(f"Valid gaze points: {filtered_count}/{original_count} ({stats['filtered_percentage']:.1f}%)")
        
        if filtered_count < self.config['min_valid_points']:
            return None, {'error': f'Insufficient valid points: {filtered_count} < {self.config["min_valid_points"]}'}
        
        return valid_df, stats
    
    def create_heatmap_visualization(self, df, subject_name, output_path):
        """
        Create a 2D heatmap visualization.
        
        Args:
            df: Filtered gaze data
            subject_name (str): Subject identifier
            output_path (str): Path to save the heatmap
            
        Returns:
            bool: Success status
        """
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
            
            # Create 2D histogram
            x = df['transformed_gaze_x'].values
            y = df['transformed_gaze_y'].values
            
            # Create heatmap - FIXED: Remove transpose to fix X/Y axis swap
            heatmap, xedges, yedges = np.histogram2d(
                x, y, bins=self.config['heatmap_bins']
            )
            
            # Apply Gaussian smoothing for better visualization
            if self.config['gaussian_sigma'] > 0:
                heatmap = gaussian_filter(heatmap, sigma=self.config['gaussian_sigma'])
            
            # Plot heatmap - Use origin='upper' for screen coordinates
            im = ax.imshow(
                heatmap.T,
                origin='upper',
                extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]],  # FIXED: Swapped y-extent for upper origin
                cmap=self.config['color_scheme'],
                aspect='equal'
            )
            
            # Customize plot
            if self.config['show_title']:
                ax.set_title(f'Gaze Heatmap - {subject_name}', fontsize=16, fontweight='bold')
            
            ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
            ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
            
            # Add colorbar
            if self.config['show_colorbar']:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Gaze Density', fontsize=12)
            
            # Add statistics overlay
            if self.config['show_stats_overlay']:
                stats_text = f'Valid Points: {len(df):,}\nMean: ({x.mean():.1f}, {y.mean():.1f})'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"Saved heatmap: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            return False
    
    def create_scatter_visualization(self, df, subject_name, output_path):
        """
        Create a scatter plot with density visualization.
        
        Args:
            df: Filtered gaze data
            subject_name (str): Subject identifier
            output_path (str): Path to save the scatter plot
            
        Returns:
            bool: Success status
        """
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
            
            x = df['transformed_gaze_x'].values
            y = df['transformed_gaze_y'].values
            
            # Create scatter plot with density colors
            scatter = ax.scatter(x, y, c=range(len(x)), cmap=self.config['color_scheme'],
                               alpha=0.6, s=1, rasterized=True)
            
            # Invert Y-axis to match screen coordinates (Y=0 at top)
            ax.invert_yaxis()
            
            # Customize plot
            if self.config['show_title']:
                ax.set_title(f'Gaze Scatter Plot - {subject_name}', fontsize=16, fontweight='bold')
            
            ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
            ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
            ax.set_aspect('equal')
            
            # Add colorbar for temporal progression
            if self.config['show_colorbar']:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Temporal Progression', fontsize=12)
            
            # Add statistics overlay
            if self.config['show_stats_overlay']:
                stats_text = f'Points: {len(df):,}\nRange X: {x.min():.0f}-{x.max():.0f}\nRange Y: {y.min():.0f}-{y.max():.0f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"Saved scatter plot: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
            return False

    def create_contour_visualization(self, df, subject_name, output_path):
        """
        Create a contour plot visualization.
        
        Args:
            df: Filtered gaze data
            subject_name (str): Subject identifier
            output_path (str): Path to save the contour plot
            
        Returns:
            bool: Success status
        """
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
            
            x = df['transformed_gaze_x'].values
            y = df['transformed_gaze_y'].values
            
            # Create 2D histogram for contour base
            heatmap, xedges, yedges = np.histogram2d(
                x, y, bins=self.config['heatmap_bins']
            )
            
            # Apply smoothing
            if self.config['gaussian_sigma'] > 0:
                heatmap = gaussian_filter(heatmap, sigma=self.config['gaussian_sigma'])
            
            # Create coordinate grids for contour
            X = (xedges[:-1] + xedges[1:]) / 2
            Y = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(X, Y)
            
            # Create contour plot - FIXED: Remove transpose to fix X/Y axis swap
            contour = ax.contour(X, Y, heatmap, levels=10, cmap=self.config['color_scheme'])
            contourf = ax.contourf(X, Y, heatmap, levels=20, cmap=self.config['color_scheme'], alpha=0.7)
            
            # Invert Y-axis to match screen coordinates (Y=0 at top)
            ax.invert_yaxis()
            
            # Customize plot
            if self.config['show_title']:
                ax.set_title(f'Gaze Contour Map - {subject_name}', fontsize=16, fontweight='bold')
            
            ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
            ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
            ax.set_aspect('equal')
            
            # Add colorbar
            if self.config['show_colorbar']:
                cbar = plt.colorbar(contourf, ax=ax)
                cbar.set_label('Gaze Density', fontsize=12)
            
            # Add contour labels
            ax.clabel(contour, inline=True, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"Saved contour plot: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating contour plot: {e}")
            return False

    def create_combined_visualization(self, df, subject_name, output_path):
        """
        Create a combined visualization with marginal density plots.
        
        Args:
            df: Filtered gaze data
            subject_name (str): Subject identifier
            output_path (str): Path to save the combined plot
            
        Returns:
            bool: Success status
        """
        try:
            # Define fixed coordinate space (1000x606 pixels)
            FIXED_X_MIN, FIXED_X_MAX = 0, 1000
            FIXED_Y_MIN, FIXED_Y_MAX = 0, 606
            
            # Set up the figure with custom grid layout (removed bottom row for stats)
            fig = plt.figure(figsize=(12, 8), dpi=self.config['dpi'])
            
            # Create a simplified 2x2 grid layout: main plot with marginal density plots only
            gs = fig.add_gridspec(2, 2,
                                 width_ratios=[4, 1],
                                 height_ratios=[1, 4],
                                 hspace=0.02, wspace=0.02)
            
            # Define subplot positions with shared axes for perfect alignment
            ax_main = fig.add_subplot(gs[1, 0])           # Main plot (bottom-left)
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)    # Top marginal (X density)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)  # Right marginal (Y density)
            
            # Hide unused corner
            fig.add_subplot(gs[0, 1]).axis('off')    # Top-right corner
            
            # Get data
            x = df['transformed_gaze_x'].values
            y = df['transformed_gaze_y'].values
            
            # === SET FIXED COORDINATE SPACE WITH PROPER ASPECT RATIO ===
            ax_main.set_xlim(FIXED_X_MIN, FIXED_X_MAX)
            ax_main.set_ylim(FIXED_Y_MAX, FIXED_Y_MIN)  # Inverted for screen coordinates (Y=0 at top)
            
            # Force equal aspect ratio to maintain 1000x606 proportions
            ax_main.set_aspect('equal', adjustable='box')
            
            # === CREATE HISTOGRAM DATA WITH FIXED BINS ===
            # Create fixed bin edges for consistent 1000x606 space
            x_bins = np.linspace(FIXED_X_MIN, FIXED_X_MAX, self.config['heatmap_bins'] + 1)
            y_bins = np.linspace(FIXED_Y_MIN, FIXED_Y_MAX, self.config['heatmap_bins'] + 1)
            
            # Create 2D histogram using fixed bins
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
            if self.config['gaussian_sigma'] > 0:
                heatmap = gaussian_filter(heatmap, sigma=self.config['gaussian_sigma'])
            
            # Create 1D histograms using the same fixed bin edges for perfect alignment
            x_hist, _ = np.histogram(x, bins=x_bins)
            y_hist, _ = np.histogram(y, bins=y_bins)
            
            # Calculate bin centers for plotting
            x_centers = (xedges[:-1] + xedges[1:]) / 2
            y_centers = (yedges[:-1] + yedges[1:]) / 2
            
            # === MAIN PLOT (Heatmap + Scatter overlay) ===
            # Plot heatmap as base layer with fixed extent
            im = ax_main.imshow(heatmap.T, origin='upper',
                               extent=[FIXED_X_MIN, FIXED_X_MAX, FIXED_Y_MAX, FIXED_Y_MIN],
                               cmap=self.config['color_scheme'], alpha=0.8)
            
            # Overlay scatter plot for individual points (sample if too many)
            if len(x) > 5000:  # Sample for performance if too many points
                sample_idx = np.random.choice(len(x), 5000, replace=False)
                x_sample, y_sample = x[sample_idx], y[sample_idx]
            else:
                x_sample, y_sample = x, y
                
            ax_main.scatter(x_sample, y_sample, c='white', s=0.5, alpha=0.3, rasterized=True)
            
            # Main plot formatting
            ax_main.set_xlabel('X Coordinate (pixels)', fontsize=12)
            ax_main.set_ylabel('Y Coordinate (pixels)', fontsize=12)
            
            # === TOP MARGINAL PLOT (X histogram) ===
            # Use bar plot to create histogram that perfectly aligns with main plot
            bin_width = x_bins[1] - x_bins[0]
            ax_top.bar(x_centers, x_hist, width=bin_width*0.9,
                      color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
            
            # Formatting for top marginal - remove labels and ticks for cleaner look
            ax_top.set_ylabel('Count', fontsize=10)
            ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax_top.tick_params(axis='y', labelsize=8)
            
            # Clean up spines
            ax_top.spines['bottom'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.spines['top'].set_visible(False)
            
            # === RIGHT MARGINAL PLOT (Y histogram) ===
            # Use barh (horizontal bar) plot for Y histogram
            bin_height = y_bins[1] - y_bins[0]
            ax_right.barh(y_centers, y_hist, height=bin_height*0.9,
                         color='darkred', alpha=0.7, edgecolor='white', linewidth=0.5)
            
            # Formatting for right marginal - remove labels and ticks for cleaner look
            ax_right.set_xlabel('Count', fontsize=10)
            ax_right.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax_right.tick_params(axis='x', labelsize=8)
            
            # Clean up spines
            ax_right.spines['left'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            ax_right.spines['top'].set_visible(False)
            
            # === ADD COLORBAR (positioned to not interfere with alignment) ===
            if self.config['show_colorbar']:
                # Create colorbar with careful positioning
                cbar = plt.colorbar(im, ax=ax_main, shrink=0.8, pad=0.15)
                cbar.set_label('Gaze Density', fontsize=10)
            
            # === ADD MINIMAL STATISTICS AS TITLE ===
            # Calculate key statistics for title
            valid_points = len(df)
            coverage_x = np.sum((x >= FIXED_X_MIN) & (x <= FIXED_X_MAX))
            coverage_y = np.sum((y >= FIXED_Y_MIN) & (y <= FIXED_Y_MAX))
            both_within = np.sum((x >= FIXED_X_MIN) & (x <= FIXED_X_MAX) &
                                (y >= FIXED_Y_MIN) & (y <= FIXED_Y_MAX))
            coverage_percent = (both_within / len(x) * 100) if len(x) > 0 else 0
            
            # Add title with key statistics
            title_text = (f'Gaze Distribution Dashboard - {subject_name}\n'
                         f'{valid_points:,} samples | {coverage_percent:.1f}% within 1000×606px bounds | '
                         f'Mean: ({x.mean():.0f}, {y.mean():.0f})px')
            
            fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.95)
            
            # Ensure tight layout with proper spacing
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)  # Make room for title
            
            # Save the plot
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"Saved aligned dashboard: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating combined visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_subject(self, csv_path, output_dir, subject_name=None):
        """
        Analyze a single subject's gaze data and create visualizations.
        
        Args:
            csv_path (str): Path to the subject's final gaze CSV
            output_dir (str): Directory to save visualizations
            subject_name (str): Subject identifier (auto-detected if None)
            
        Returns:
            dict: Analysis results and statistics
        """
        csv_path = Path(csv_path)
        output_dir = Path(output_dir)
        
        # Auto-detect subject name if not provided
        if subject_name is None:
            subject_name = csv_path.stem.replace('_final_gaze_data', '')
        
        print(f"\nAnalyzing gaze data for: {subject_name}")
        print(f"Input CSV: {csv_path}")
        print(f"Output directory: {output_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and filter data
        df = self.load_gaze_data(csv_path)
        if df is None:
            return {'success': False, 'error': 'Failed to load data'}
        
        filtered_df, stats = self.filter_valid_gaze_data(df)
        if filtered_df is None:
            return {'success': False, 'error': stats.get('error', 'No valid data')}
        
        # Initialize results
        results = {
            'success': True,
            'subject_name': subject_name,
            'input_csv': str(csv_path),
            'output_dir': str(output_dir),
            'statistics': stats,
            'visualizations_created': []
        }
        
        # Create visualizations
        format_ext = self.config['output_format']
        
        if self.config['create_heatmap']:
            heatmap_path = output_dir / f"{subject_name}_heatmap.{format_ext}"
            if self.create_heatmap_visualization(filtered_df, subject_name, heatmap_path):
                results['visualizations_created'].append(str(heatmap_path))
        
        if self.config['create_scatter']:
            scatter_path = output_dir / f"{subject_name}_scatter.{format_ext}"
            if self.create_scatter_visualization(filtered_df, subject_name, scatter_path):
                results['visualizations_created'].append(str(scatter_path))
        
        if self.config['create_contour']:
            contour_path = output_dir / f"{subject_name}_contour.{format_ext}"
            if self.create_contour_visualization(filtered_df, subject_name, contour_path):
                results['visualizations_created'].append(str(contour_path))
        
        if self.config['create_combined']:
            combined_path = output_dir / f"{subject_name}_dashboard.{format_ext}"
            if self.create_combined_visualization(filtered_df, subject_name, combined_path):
                results['visualizations_created'].append(str(combined_path))
        
        # Save statistics if requested
        if self.config['save_stats']:
            stats_path = output_dir / f"{subject_name}_gaze_statistics.json"
            try:
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                print(f"Saved statistics: {stats_path}")
            except Exception as e:
                print(f"Could not save statistics: {e}")
        
        print(f"Analysis complete: {len(results['visualizations_created'])} visualizations created")
        return results
