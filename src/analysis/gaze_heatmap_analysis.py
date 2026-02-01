#!/usr/bin/env python3
"""
gaze_heatmap_analysis.py - Generate 2D gaze heatmaps for visual analysis

This script takes the processed gaze CSV and creates visual representations of where the subject looked.
It produces:
1.  **Heatmaps**: Color-coded density plots (Hot = High attention).
2.  **Scatter Plots**: Raw points colored by time (showing the "path" of the eye).
3.  **Contour Maps**: Topographic-style maps of gaze density.
4.  **Dashboards**: A combined view with marginal histograms (X/Y distribution).
"""

# Import scipy BEFORE matplotlib/seaborn to avoid BLAS threading deadlock
from scipy import stats
from scipy.ndimage import gaussian_filter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend (safe for servers/headless machines)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import json
import traceback
from ..logging_config import get_logger

logger = get_logger(__name__)

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
        """
        # Default configuration
        self.config = {
            # Visualization settings
            'figure_size': (12, 8),
            'dpi': 300,             # High resolution for publication quality
            'color_scheme': 'viridis',
            'heatmap_bins': 50,     # Resolution: 50x50 grid
            'gaussian_sigma': 1.0,  # Smoothing: Blurs the heatmap slightly to look natural
            
            # Output settings
            'output_format': 'png',
            'save_stats': True,
            'show_title': True,
            'show_stats_overlay': True, # Show valid point count on the image
            'show_colorbar': True,
            
            # Data filtering
            'workspace_bounds': None,  # (min_x, max_x, min_y, max_y)
            'outlier_percentile': 99,  # Remove the top 1% extreme points (glitches)
            'min_valid_points': 100,   # Don't plot if we have almost no data
            
            # What to draw?
            'create_heatmap': True,
            'create_scatter': True,
            'create_contour': True,
            'create_combined': True
        }
        
        # Override defaults with user config
        if config:
            self.config.update(config)
    
    def load_gaze_data(self, csv_path):
        """
        Load and validate gaze data from CSV file.
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Check for necessary columns
            required_cols = ['gaze_timestamp', 'transformed_gaze_x', 'transformed_gaze_y']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Loaded {len(df)} gaze samples from {Path(csv_path).name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading gaze data from {csv_path}: {e}")
            return None
    
    def filter_valid_gaze_data(self, df):
        """
        Filter and clean gaze data for visualization.
        Removes NaNs and extreme outliers.
        """
        original_count = len(df)
        
        # 1. Remove NaNs (frames where gaze wasn't detected)
        valid_df = df.dropna(subset=['transformed_gaze_x', 'transformed_gaze_y'])
        valid_count = len(valid_df)
        
        if valid_count == 0:
            return None, {'error': 'No valid gaze points found'}
        
        # 2. Remove spatial outliers (optional)
        if self.config['outlier_percentile'] < 100:
            percentile = self.config['outlier_percentile']
            
            # Calculate percentile bounds
            x_lower = np.percentile(valid_df['transformed_gaze_x'], (100 - percentile) / 2)
            x_upper = np.percentile(valid_df['transformed_gaze_x'], 100 - (100 - percentile) / 2)
            y_lower = np.percentile(valid_df['transformed_gaze_y'], (100 - percentile) / 2)
            y_upper = np.percentile(valid_df['transformed_gaze_y'], 100 - (100 - percentile) / 2)
            
            # Keep only data within bounds
            valid_df = valid_df[
                (valid_df['transformed_gaze_x'] >= x_lower) &
                (valid_df['transformed_gaze_x'] <= x_upper) &
                (valid_df['transformed_gaze_y'] >= y_lower) &
                (valid_df['transformed_gaze_y'] <= y_upper)
            ]
        
        # 3. Apply manual workspace bounds (if configured)
        if self.config['workspace_bounds']:
            min_x, max_x, min_y, max_y = self.config['workspace_bounds']
            valid_df = valid_df[
                (valid_df['transformed_gaze_x'] >= min_x) &
                (valid_df['transformed_gaze_x'] <= max_x) &
                (valid_df['transformed_gaze_y'] >= min_y) &
                (valid_df['transformed_gaze_y'] <= max_y)
            ]
        
        filtered_count = len(valid_df)
        
        # Statistics for reporting
        stats = {
            'original_samples': original_count,
            'valid_samples': valid_count,
            'filtered_samples': filtered_count,
            'valid_percentage': (valid_count / original_count * 100) if original_count > 0 else 0,
            'filtered_percentage': (filtered_count / original_count * 100) if original_count > 0 else 0,
            'x_mean': valid_df['transformed_gaze_x'].mean() if filtered_count > 0 else 0,
            'y_mean': valid_df['transformed_gaze_y'].mean() if filtered_count > 0 else 0
        }
        
        logger.info(f"Valid gaze points: {filtered_count}/{original_count} ({stats['filtered_percentage']:.1f}%)")
        
        if filtered_count < self.config['min_valid_points']:
            return None, {'error': f'Insufficient valid points: {filtered_count} < {self.config["min_valid_points"]}'}
        
        return valid_df, stats
    
    def create_heatmap_visualization(self, df, subject_name, output_path):
        """
        Create a 2D heatmap visualization using a histogram.
        """
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
            
            x = df['transformed_gaze_x'].values
            y = df['transformed_gaze_y'].values
            
            # Create 2D histogram (binning points into a grid)
            heatmap, xedges, yedges = np.histogram2d(
                x, y, bins=self.config['heatmap_bins']
            )
            
            # Apply Gaussian smoothing to make it look like a heat map
            if self.config['gaussian_sigma'] > 0:
                heatmap = gaussian_filter(heatmap, sigma=self.config['gaussian_sigma'])
            
            # Plot
            im = ax.imshow(
                heatmap.T,
                origin='upper',
                extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]],
                cmap=self.config['color_scheme'],
                aspect='equal'
            )
            
            if self.config['show_title']:
                ax.set_title(f'Gaze Heatmap - {subject_name}', fontsize=16, fontweight='bold')
            
            ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
            ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
            
            if self.config['show_colorbar']:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Gaze Density', fontsize=12)
            
            if self.config['show_stats_overlay']:
                stats_text = f'Valid Points: {len(df):,}\nMean: ({x.mean():.1f}, {y.mean():.1f})'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved heatmap: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return False
    
    def create_scatter_visualization(self, df, subject_name, output_path):
        """
        Create a scatter plot where points are colored by order of occurrence (time).
        """
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
            
            x = df['transformed_gaze_x'].values
            y = df['transformed_gaze_y'].values
            
            # Scatter plot with color mapping based on index (time)
            # This creates a gradient from start (Purple) to end (Yellow)
            scatter = ax.scatter(x, y, c=range(len(x)), cmap=self.config['color_scheme'],
                               alpha=0.6, s=1, rasterized=True)
            
            # Invert Y-axis so (0,0) is top-left (screen coordinates)
            ax.invert_yaxis()
            
            if self.config['show_title']:
                ax.set_title(f'Gaze Scatter Plot - {subject_name}', fontsize=16, fontweight='bold')
            
            ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
            ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
            ax.set_aspect('equal')
            
            if self.config['show_colorbar']:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Temporal Progression', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved scatter plot: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return False

    def create_contour_visualization(self, df, subject_name, output_path):
        """
        Create a topographic contour map of gaze density.
        """
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
            
            x = df['transformed_gaze_x'].values
            y = df['transformed_gaze_y'].values
            
            heatmap, xedges, yedges = np.histogram2d(
                x, y, bins=self.config['heatmap_bins']
            )
            
            if self.config['gaussian_sigma'] > 0:
                heatmap = gaussian_filter(heatmap, sigma=self.config['gaussian_sigma'])
            
            # Grid for contouring
            X = (xedges[:-1] + xedges[1:]) / 2
            Y = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(X, Y)
            
            # Filled contours
            contourf = ax.contourf(X, Y, heatmap, levels=20, cmap=self.config['color_scheme'], alpha=0.7)
            
            ax.invert_yaxis()
            
            if self.config['show_title']:
                ax.set_title(f'Gaze Contour Map - {subject_name}', fontsize=16, fontweight='bold')
            
            ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
            ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
            ax.set_aspect('equal')
            
            if self.config['show_colorbar']:
                cbar = plt.colorbar(contourf, ax=ax)
                cbar.set_label('Gaze Density', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved contour plot: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating contour plot: {e}")
            return False

    def create_combined_visualization(self, df, subject_name, output_path):
        """
        Create a dashboard with:
        - Central Heatmap/Scatter plot
        - Top Marginal Histogram (X distribution)
        - Right Marginal Histogram (Y distribution)
        """
        try:
            FIXED_X_MIN, FIXED_X_MAX = 0, 1000
            FIXED_Y_MIN, FIXED_Y_MAX = 0, 606
            
            # Setup complex grid layout
            fig = plt.figure(figsize=(14, 8), dpi=self.config['dpi'])
            gs = fig.add_gridspec(2, 3, width_ratios=[1, 4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
            
            ax_main = fig.add_subplot(gs[1, 1])
            ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
            ax_cbar = fig.add_subplot(gs[1, 0])
            
            # Hide unused axes
            fig.add_subplot(gs[0, 0]).axis('off')
            fig.add_subplot(gs[0, 2]).axis('off')
            
            x = df['transformed_gaze_x'].values
            y = df['transformed_gaze_y'].values
            
            # Set fixed bounds
            ax_main.set_xlim(FIXED_X_MIN, FIXED_X_MAX)
            ax_main.set_ylim(FIXED_Y_MAX, FIXED_Y_MIN) # Inverted Y
            ax_main.set_aspect('equal', adjustable='box')
            
            # Bins
            x_bins = np.linspace(FIXED_X_MIN, FIXED_X_MAX, self.config['heatmap_bins'] + 1)
            y_bins = np.linspace(FIXED_Y_MIN, FIXED_Y_MAX, self.config['heatmap_bins'] + 1)
            
            heatmap, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
            if self.config['gaussian_sigma'] > 0:
                heatmap = gaussian_filter(heatmap, sigma=self.config['gaussian_sigma'])
            
            # Marginal Histograms
            x_hist, _ = np.histogram(x, bins=x_bins)
            y_hist, _ = np.histogram(y, bins=y_bins)
            
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2
            
            # --- Draw Main Plot ---
            im = ax_main.imshow(heatmap.T, origin='upper',
                               extent=[FIXED_X_MIN, FIXED_X_MAX, FIXED_Y_MAX, FIXED_Y_MIN],
                               cmap=self.config['color_scheme'], alpha=0.8)
            
            # Overlay scatter points (downsample if too many)
            if len(x) > 5000:
                sample_idx = np.random.choice(len(x), 5000, replace=False)
                x_sample, y_sample = x[sample_idx], y[sample_idx]
            else:
                x_sample, y_sample = x, y
                
            ax_main.scatter(x_sample, y_sample, c='white', s=0.5, alpha=0.3, rasterized=True)
            
            ax_main.set_xlabel('X Coordinate (pixels)', fontsize=12)
            ax_main.set_ylabel('Y Coordinate (pixels)', fontsize=12)
            
            # --- Draw Marginal Plots ---
            # Top (X)
            bin_width = x_bins[1] - x_bins[0]
            ax_top.bar(x_centers, x_hist, width=bin_width*0.9, color='steelblue', alpha=0.7)
            ax_top.set_ylabel('Count', fontsize=10)
            ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.spines['bottom'].set_visible(False)
            
            # Right (Y)
            bin_height = y_bins[1] - y_bins[0]
            ax_right.barh(y_centers, y_hist, height=bin_height*0.9, color='darkred', alpha=0.7)
            ax_right.set_xlabel('Count', fontsize=10)
            ax_right.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)
            
            # Colorbar
            if self.config['show_colorbar']:
                cbar = plt.colorbar(im, cax=ax_cbar)
                cbar.set_label('Gaze Density', fontsize=10)
            else:
                ax_cbar.axis('off')
            
            # Title
            valid_points = len(df)
            coverage_x = np.sum((x >= FIXED_X_MIN) & (x <= FIXED_X_MAX))
            coverage_y = np.sum((y >= FIXED_Y_MIN) & (y <= FIXED_Y_MAX))
            coverage_percent = (coverage_x / len(x) * 100) # Approximate
            
            title_text = (f'Gaze Distribution Dashboard - {subject_name}\n'
                         f'{valid_points:,} samples | Mean: ({x.mean():.0f}, {y.mean():.0f})px')
            
            fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.95)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved aligned dashboard: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating combined visualization: {e}")
            return False
    
    def analyze_subject(self, csv_path, output_dir, subject_name=None):
        """
        Main runner function for a single subject.
        """
        csv_path = Path(csv_path)
        output_dir = Path(output_dir)
        
        if subject_name is None:
            subject_name = csv_path.stem.replace('_final_gaze_data', '')
        
        logger.info(f"Analyzing gaze data for: {subject_name}")
        
        # Load
        df = self.load_gaze_data(csv_path)
        if df is None: return {'success': False, 'error': 'Failed to load data'}
        
        # Filter
        filtered_df, stats = self.filter_valid_gaze_data(df)
        if filtered_df is None: return {'success': False, 'error': stats.get('error', 'No valid data')}
        
        results = {
            'success': True,
            'subject_name': subject_name,
            'statistics': stats,
            'visualizations_created': []
        }
        
        # Create Plots
        format_ext = self.config['output_format']
        
        if self.config['create_heatmap']:
            path = output_dir / f"{subject_name}_heatmap.{format_ext}"
            if self.create_heatmap_visualization(filtered_df, subject_name, path):
                results['visualizations_created'].append(str(path))
        
        if self.config['create_scatter']:
            path = output_dir / f"{subject_name}_scatter.{format_ext}"
            if self.create_scatter_visualization(filtered_df, subject_name, path):
                results['visualizations_created'].append(str(path))
        
        if self.config['create_contour']:
            path = output_dir / f"{subject_name}_contour.{format_ext}"
            if self.create_contour_visualization(filtered_df, subject_name, path):
                results['visualizations_created'].append(str(path))
        
        if self.config['create_combined']:
            path = output_dir / f"{subject_name}_dashboard.{format_ext}"
            if self.create_combined_visualization(filtered_df, subject_name, path):
                results['visualizations_created'].append(str(path))
        
        return results
