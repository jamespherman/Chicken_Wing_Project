#!/usr/bin/env python3
"""
batch_process_with_heatmaps.py - Enhanced master script with automatic heatmap generation and skip list

This script is the main entry point for the analysis pipeline. It automates the processing
of multiple subjects found in the data directory.

The pipeline performs 4 main steps for each subject:
1.  **Perspective Correction**: Stabilizes the video and gaze data using ArUco markers.
2.  **CSV Generation**: Creates a high-resolution dataset with physics-based metrics (velocity, pupil, IMU).
3.  **Visualizations**: Generates heatmaps and scatter plots of the gaze patterns.
4.  **Clinical Analysis**: Calculates skill metrics (Goals A-D) and generates clinical dashboards.

Usage:
    python3 src/batch_process_with_heatmaps.py
"""

import os
import sys
import time
import gc
from pathlib import Path
from datetime import datetime
import json
import traceback

from .logging_config import configure_logging, get_logger

# Configure logging to print progress to the console and save to files
logger = get_logger(__name__)


# Import the processing modules from other files in the src/ directory
try:
    # Step 1: Perspective Correction logic
    from .processing.gaze_on_perspective_corrected_frames_refactored import process_gaze_with_perspective_correction
    # Step 2: CSV Creation logic
    from .processing.create_final_csv_refactored import create_final_gaze_csv
    # Step 3: Heatmap Generation logic
    from .analysis.gaze_heatmap_analysis import GazeHeatmapAnalyzer
    # Step 4: Clinical Analysis logic
    from .analysis.whole_session_analysis import WholeSessionAnalyzer
    # Step 5: Clinical Visualization logic
    from .analysis.visualizations import GazeVisualizer

    # Utilities for finding folders and creating reports
    from .processing.batch_processing.subject_discovery import discover_subject_folders
    from .processing.batch_processing.reporting import save_processing_log, create_summary_report
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please ensure all required modules are available.")
    sys.exit(1)


class EnhancedBatchProcessor:
    """
    This class manages the entire batch processing workflow.
    It reads configuration, finds data, and runs the pipeline for each subject.
    """
    
    def __init__(self, config=None):
        """
        Initialize the processor.
        
        Args:
            config (dict): Optional configuration to override defaults.
        """
        # --- Find project paths automatically ---
        # Get the path of this script to locate project root
        script_path = Path(__file__).resolve()
        src_dir = script_path.parent
        self.project_root = src_dir.parent

        # Load configuration from config.json
        self.config = self._load_base_config()

        # Set input/output directories relative to project root
        self.config['input_base_dir'] = self.project_root / self.config.get('input_base_dir', 'data/raw')
        self.config['output_base_dir'] = self.project_root

        # Apply user overrides if provided
        if config:
            self._update_config_recursive(self.config, config)

        # Initialize counters to track progress
        self.results = []
        self.start_time = None
        self.total_subjects = 0
        self.successful_subjects = 0
        self.failed_subjects = 0
        self.skipped_subjects = 0

        # Create the folder structure for outputs (reports/, data/processed/, etc.)
        self._create_output_directories()

        # Initialize the Heatmap Analyzer (for Step 3)
        if self.config['generate_heatmaps']:
            self.heatmap_analyzer = GazeHeatmapAnalyzer(self.config['heatmap_config'])
        else:
            self.heatmap_analyzer = None

        # Initialize the Visualizer (for Step 5 - Clinical Dashboards)
        self.gaze_visualizer = GazeVisualizer(self.config.get('visualization_config', {}))

    def _load_base_config(self):
        """
        Load 'config.json' from the project root.
        If keys are missing, it uses the default values defined here.
        """
        config_path = self.project_root / 'config.json'

        # Load the file
        with open(config_path, 'r') as f:
            user_config = json.load(f)

        # Define default settings (fallback values)
        default_config = {
            'video_filename': 'scenevideo.mp4',
            'gaze_filename': 'gazedata.gz',
            'subject_folder_pattern': '*',
            'subjects_to_skip': [],
            'output_width': 1000,   # Width of stabilized video
            'output_height': 606,   # Height of stabilized video
            'target_markers': [13, 14, 15, 16], # IDs of ArUco markers to look for
            'frame_width': 1920,    # Original video width
            'frame_height': 1080,   # Original video height
            'processing_options': {
                'use_preselected_parameters': False,
                'use_frame_preprocessing': False, # Enhance contrast if markers are hard to find
                'use_outer_points': False,
                'show_video': False     # Don't show video window during batch processing
            },
            'skip_existing': True,      # Don't re-process if output files exist
            'create_summary_report': True,
            'generate_heatmaps': True,
            'heatmap_config': {
                'figure_size': (12, 8),
                'dpi': 300,
                'color_scheme': 'viridis',
                'heatmap_bins': 50,
                'gaussian_sigma': 1.0,  # Smoothing factor
                'output_format': 'png',
                'create_heatmap': True,
                'create_scatter': True,
                'create_contour': True,
                'create_combined': True,
                'show_stats_overlay': True,
                'save_stats': True,
                'min_valid_points': 100
            },
            'run_whole_session_analysis': True, # Enable Goals A-D analysis
            'whole_session_config': {
                'include_goal_d': False  # Visual strategy requires video processing (slow)
            }
        }
        
        # Merge user config into defaults
        self._update_config_recursive(default_config, user_config)
        return default_config

    def _update_config_recursive(self, base_dict, new_dict):
        """
        Helper to merge nested dictionaries (configuration).
        """
        for key, value in new_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_config_recursive(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _create_output_directories(self):
        """
        Create the standard directory structure for outputs.
        """
        base_dir = Path(self.config['output_base_dir']).resolve()
        
        self.output_root = base_dir
        self.reports_dir = self.output_root / "reports"
        self.data_dir = self.output_root / "data"

        self.figures_dir = self.reports_dir / "figures"
        self.logs_dir = self.reports_dir / "logs"
        self.processed_data_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory structure created at: {self.output_root}")
    
    def create_output_paths(self, subject_folder):
        """
        Define the file paths where outputs will be saved for a specific subject.
        """
        subject_name = subject_folder.name
        
        subject_data_dir = self.processed_data_dir / subject_name
        subject_data_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {
            'output_dir': subject_data_dir,
            # Step 1 Output
            'corrected_video': subject_data_dir / f"{subject_name}_gaze_corrected_video.mp4",
            'intermediate_csv': subject_data_dir / f"{subject_name}_gaze_output.csv",
            'transformation_history': subject_data_dir / f"{subject_name}_transformation_history.npy",
            # Step 2 Output (The Golden Source)
            'final_csv': subject_data_dir / f"{subject_name}_final_gaze_data.csv",
            # Step 3 Output (Heatmaps)
            'heatmap_png': self.figures_dir / f"{subject_name}_heatmap.png",
            'scatter_png': self.figures_dir / f"{subject_name}_scatter.png",
            'contour_png': self.figures_dir / f"{subject_name}_contour.png",
            'dashboard_png': self.figures_dir / f"{subject_name}_dashboard.png",
            # Step 4 Output (Analysis Data)
            'processing_log': self.logs_dir / f"{subject_name}_processing_log.txt",
            'gaze_stats': self.logs_dir / f"{subject_name}_gaze_statistics.json",
            'whole_session_analysis': self.logs_dir / f"{subject_name}_whole_session_analysis.json",
            # Step 5 Output (Clinical Viz)
            'viz_cognitive_fingerprint': self.figures_dir / f"{subject_name}_viz_cognitive_fingerprint.png",
            'viz_main_sequence': self.figures_dir / f"{subject_name}_viz_main_sequence.png",
            'viz_stress_timeline': self.figures_dir / f"{subject_name}_viz_stress_timeline.png",
            'viz_stability_radar': self.figures_dir / f"{subject_name}_viz_stability_radar.png",
        }

        return outputs
    
    def check_existing_outputs(self, output_paths):
        """
        Check if we can skip this subject because all files already exist.
        """
        if not self.config['skip_existing']:
            return False

        final_csv = output_paths['final_csv']
        dashboard_png = output_paths['dashboard_png']

        # We generally need at least the final CSV
        if not final_csv.exists():
            return False

        # If heatmaps are enabled, check for them
        if self.config['generate_heatmaps'] and not dashboard_png.exists():
            return False

        # If analysis is enabled, check for clinical charts
        if self.config.get('run_whole_session_analysis', True):
            viz_files = [
                output_paths['viz_cognitive_fingerprint'],
                output_paths['viz_main_sequence'],
                output_paths['viz_stress_timeline'],
                output_paths['viz_stability_radar'],
            ]
            if not all(f.exists() for f in viz_files):
                return False

        logger.info(f"Skipping: Outputs already exist")
        return True
    
    def process_single_subject(self, subject_folder):
        """
        Run the full pipeline for ONE subject.
        
        Args:
            subject_folder (Path): Directory containing raw data for one subject.
            
        Returns:
            dict: Summary of results for this subject.
        """
        subject_name = subject_folder.name
        logger.info(f"Processing subject: {subject_name}")
        
        output_paths = self.create_output_paths(subject_folder)
        
        # Check for skip
        if self.check_existing_outputs(output_paths):
            return {
                'subject_name': subject_name,
                'status': 'skipped',
                'reason': 'Outputs already exist',
                'processing_time': 0
            }
        
        subject_start_time = time.time()
        result = {
            'subject_name': subject_name,
            'subject_folder': str(subject_folder),
            'status': 'failed',
            'error_message': None,
            'processing_time': 0
        }
        
        try:
            # === STEP 1: Perspective Correction ===
            # Detects ArUco markers and stabilizes the video/gaze
            transformation_exists = output_paths['transformation_history'].exists()
            if transformation_exists:
                logger.info(f"Step 1: Skipping (transformation history exists)")
                result['step1_stats'] = {'skipped': True}
            else:
                logger.info(f"Step 1: Processing video with gaze data...")

                step1_stats = process_gaze_with_perspective_correction(
                    video_path=str(subject_folder / self.config['video_filename']),
                    gaze_file_path=str(subject_folder / self.config['gaze_filename']),
                    output_video_path=str(output_paths['corrected_video']),
                    csv_output_path=str(output_paths['intermediate_csv']),
                    transformation_history_path=str(output_paths['transformation_history']),
                    output_width=self.config['output_width'],
                    output_height=self.config['output_height'],
                    target_markers=self.config['target_markers'],
                    **self.config['processing_options']
                )

                result['step1_stats'] = step1_stats
                logger.info(f"Step 1 completed: {step1_stats['frames_with_valid_homography']} frames stabilized")

            # === STEP 2: Final CSV Generation ===
            # Merges gaze, IMU, and video data into one high-res CSV with physics metrics
            final_csv_exists = output_paths['final_csv'].exists()
            if final_csv_exists:
                logger.info(f"Step 2: Skipping (final CSV exists)")
                result['step2_stats'] = {'skipped': True, 'success': True}
            else:
                logger.info(f"Step 2: Creating final high-resolution gaze CSV...")

                step2_stats = create_final_gaze_csv(
                    gaze_file_path=str(subject_folder / self.config['gaze_filename']),
                    transformation_history_path=str(output_paths['transformation_history']),
                    output_csv_path=str(output_paths['final_csv']),
                    frame_width=self.config['frame_width'],
                    frame_height=self.config['frame_height']
                )

                if not step2_stats or not step2_stats.get('success', False):
                    result['error_message'] = "Step 2 failed: Could not create final CSV"
                    return result

                result['step2_stats'] = step2_stats
                logger.info(f"Step 2 completed: CSV created")
            
            # === STEP 3: Heatmap Generation ===
            # Creates visual maps of where the surgeon looked
            if self.config['generate_heatmaps'] and self.heatmap_analyzer:
                logger.info(f"Step 3: Generating gaze heatmap visualizations...")
                
                step3_stats = self.heatmap_analyzer.analyze_subject(
                    csv_path=str(output_paths['final_csv']),
                    output_dir=str(self.figures_dir),
                    subject_name=subject_name
                )
                
                # Save statistics JSON
                if step3_stats.get('success', False) and step3_stats.get('statistics'):
                    with open(output_paths['gaze_stats'], 'w') as f:
                        json.dump(step3_stats['statistics'], f, indent=2, default=str)
                
                result['step3_stats'] = step3_stats

            # === STEP 4: Scientific Analysis (Goals A-D) ===
            # Calculates skill metrics like fixation rate, cognitive load, etc.
            analyzer = None
            if self.config.get('run_whole_session_analysis', True):
                logger.info(f"Step 4: Running whole-session analysis...")

                try:
                    analyzer = WholeSessionAnalyzer()
                    ws_config = self.config.get('whole_session_config', {})

                    step4_stats = analyzer.run_complete_analysis(
                        csv_path=str(output_paths['final_csv']),
                        video_path=str(subject_folder / self.config['video_filename']),
                        include_goal_d=ws_config.get('include_goal_d', False)
                    )

                    # Save analysis results to JSON
                    with open(output_paths['whole_session_analysis'], 'w') as f:
                        # Helper to handle numpy types in JSON
                        def convert_to_serializable(obj):
                            if isinstance(obj, dict):
                                return {k: convert_to_serializable(v) for k, v in obj.items()}
                            elif isinstance(obj, (list, tuple)):
                                return [convert_to_serializable(i) for i in obj]
                            elif hasattr(obj, 'item'):  # numpy scalar
                                return obj.item()
                            elif obj is None or isinstance(obj, (int, float, str, bool)):
                                return obj
                            else:
                                return str(obj)

                        json.dump(convert_to_serializable(step4_stats), f, indent=2)

                    result['step4_stats'] = step4_stats
                    logger.info(f"Step 4 completed: Analysis saved")

                except Exception as e:
                    logger.warning(f"Step 4 warning: {e}")
                    result['step4_stats'] = {'error': str(e)}
                    analyzer = None

            # === STEP 5: Clinical Dashboards ===
            # Generates the Radar Charts and Stress Timelines
            if self.config.get('run_whole_session_analysis', True) and analyzer is not None:
                logger.info(f"Step 5: Generating clinical visualizations...")

                try:
                    step5_stats = self.gaze_visualizer.create_all_visualizations(
                        analyzer=analyzer,
                        output_dir=str(self.figures_dir),
                        subject_name=subject_name
                    )
                    result['step5_stats'] = step5_stats
                    logger.info(f"Step 5 completed")

                except Exception as e:
                    logger.warning(f"Step 5 warning: {e}")
                    result['step5_stats'] = {'error': str(e)}

            result['status'] = 'success'
        
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"Processing failed: {e}")
            logger.exception("Traceback:")
        
        result['processing_time'] = time.time() - subject_start_time
        save_processing_log(result, output_paths['processing_log'])
        
        return result
    
    def run(self):
        """
        Main execution method. Iterates through all found subjects.
        """
        self.start_time = datetime.now()
        logger.info("Starting batch processing...")
        
        # Discover folders
        subject_folders, self.skipped_subjects = discover_subject_folders(
            self.config['input_base_dir'],
            self.config['video_filename'],
            self.config['gaze_filename'],
            self.config['subject_folder_pattern'],
            self.config['subjects_to_skip']
        )
        
        if not subject_folders:
            logger.warning("No valid subject folders found. Exiting.")
            return {'success': False}
        
        self.total_subjects = len(subject_folders)
        
        # Process loop
        for i, subject_folder in enumerate(subject_folders, 1):
            logger.info(f"\nProcessing subject {i}/{self.total_subjects}")
            
            result = self.process_single_subject(subject_folder)
            self.results.append(result)

            if result['status'] == 'success':
                self.successful_subjects += 1
            elif result['status'] == 'failed':
                self.failed_subjects += 1

            # Clean up memory
            gc.collect()

        # Generate final summary report
        create_summary_report(
            self.config, self.results, self.skipped_subjects, self.total_subjects,
            self.successful_subjects, self.failed_subjects, self.start_time,
            self.logs_dir, self.figures_dir, self.processed_data_dir
        )
        
        logger.info(f"BATCH PROCESSING COMPLETE")
        return {
            'success': True,
            'total_subjects': self.total_subjects,
            'successful_subjects': self.successful_subjects
        }


def batch_process_subjects(config=None):
    """
    Wrapper function to run the batch processor easily.
    """
    processor = EnhancedBatchProcessor(config)
    return processor.run()


def main():
    """
    Command-line entry point.
    """
    try:
        # Load or create config
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        config_path = project_root / 'config.json'

        if not config_path.exists():
            # Create default config if missing
            default_config_data = {
                "input_base_dir": "data/raw",
                "subjects_to_skip": [],
                "generate_heatmaps": True,
                "create_summary_report": True
            }
            with open(config_path, 'w') as f:
                json.dump(default_config_data, f, indent=4)

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Run
        results = batch_process_subjects(config)
        return results.get('success', False)
            
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
