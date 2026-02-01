#!/usr/bin/env python3
"""
batch_process_with_heatmaps.py - Enhanced master script with automatic heatmap generation and skip list

This enhanced script automatically processes multiple subjects by:
1. Running gaze processing with perspective correction for each subject
2. Creating final high-resolution gaze CSV files
3. Generating heatmap visualizations
4. Organizing outputs with logical naming conventions
5. Skipping known problematic subjects using a skip list

Usage:
    python3 batch_process_with_heatmaps.py
    
Or import and use programmatically:
    from batch_process_with_heatmaps import batch_process_subjects
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

# Configure logging for the application
logger = get_logger(__name__)


# Import the refactored processing functions
try:
    from .processing.gaze_on_perspective_corrected_frames_refactored import process_gaze_with_perspective_correction
    from .processing.create_final_csv_refactored import create_final_gaze_csv
    from .analysis.gaze_heatmap_analysis import GazeHeatmapAnalyzer
    from .analysis.whole_session_analysis import WholeSessionAnalyzer
    from .analysis.visualizations import GazeVisualizer
    from .processing.batch_processing.subject_discovery import discover_subject_folders
    from .processing.batch_processing.reporting import save_processing_log, create_summary_report
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please ensure all required modules are available.")
    sys.exit(1)


class EnhancedBatchProcessor:
    """
    Enhanced batch processor class with automatic heatmap generation and skip list functionality.
    """
    
    def __init__(self, config=None):
        """
        Initialize the enhanced batch processor with configuration.
        
        Args:
            config (dict): Configuration dictionary with processing parameters
        """
        # --- Find project paths automatically ---
        script_path = Path(__file__).resolve()
        src_dir = script_path.parent
        self.project_root = src_dir.parent

        # Load base configuration from JSON file
        self.config = self._load_base_config()

        # Set dynamic paths
        self.config['input_base_dir'] = self.project_root / self.config.get('input_base_dir', 'data/raw')
        self.config['output_base_dir'] = self.project_root

        # Update with user config if provided
        if config:
            self._update_config_recursive(self.config, config)

        # Initialize tracking variables
        self.results = []
        self.start_time = None
        self.total_subjects = 0
        self.successful_subjects = 0
        self.failed_subjects = 0
        self.skipped_subjects = 0

        # Create organized output directories
        self._create_output_directories()

        # Initialize heatmap analyzer
        if self.config['generate_heatmaps']:
            self.heatmap_analyzer = GazeHeatmapAnalyzer(self.config['heatmap_config'])
        else:
            self.heatmap_analyzer = None

        # Initialize gaze visualizer for clinical visualizations
        self.gaze_visualizer = GazeVisualizer(self.config.get('visualization_config', {}))

    def _load_base_config(self):
        """
        Load base configuration from config.json and provide sane defaults.
        """
        config_path = self.project_root / 'config.json'

        with open(config_path, 'r') as f:
            user_config = json.load(f)

        # Start with sane defaults
        default_config = {
            'video_filename': 'scenevideo.mp4',
            'gaze_filename': 'gazedata.gz',
            'subject_folder_pattern': '*',
            'subjects_to_skip': [],
            'output_width': 1000,
            'output_height': 606,
            'target_markers': [13, 14, 15, 16],
            'frame_width': 1920,
            'frame_height': 1080,
            'processing_options': {
                'use_preselected_parameters': False,
                'use_frame_preprocessing': False,
                'use_outer_points': False,
                'show_video': False
            },
            'skip_existing': True,
            'create_summary_report': True,
            'generate_heatmaps': True,
            'heatmap_config': {
                'figure_size': (12, 8),
                'dpi': 300,
                'color_scheme': 'viridis',
                'heatmap_bins': 50,
                'gaussian_sigma': 1.0,
                'output_format': 'png',
                'create_heatmap': True,
                'create_scatter': True,
                'create_contour': True,
                'create_combined': True,
                'show_stats_overlay': True,
                'save_stats': True,
                'min_valid_points': 100
            },
            'run_whole_session_analysis': True,
            'whole_session_config': {
                'include_goal_d': False  # Visual strategy requires video processing
            }
        }
        
        # Update defaults with user config
        self._update_config_recursive(default_config, user_config)
        return default_config

    def _update_config_recursive(self, base_dict, new_dict):
        """
        Recursively update dictionary.
        """
        for key, value in new_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_config_recursive(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _create_output_directories(self):
        """
        Create the organized output directory structure.
        """
        # Convert output_base_dir to Path object and resolve it
        base_dir = Path(self.config['output_base_dir']).resolve()
        
        # Create main output directories with explicit paths
        self.output_root = base_dir
        self.reports_dir = self.output_root / "reports"
        self.data_dir = self.output_root / "data"

        self.figures_dir = self.reports_dir / "figures"
        self.logs_dir = self.reports_dir / "logs"
        self.processed_data_dir = self.data_dir / "processed"
        
        # Create all directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory structure created:")
        logger.info(f"  Root: {self.output_root}")
        logger.info(f"  Processed Data: {self.processed_data_dir}")
        logger.info(f"  Figures: {self.figures_dir}")
        logger.info(f"  Logs: {self.logs_dir}")
    
    def _update_config(self, user_config):
        """
        Recursively update configuration with user-provided values.
        """
        def update_dict(base_dict, new_dict):
            for key, value in new_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_dict(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_dict(self.config, user_config)
    
    
    def create_output_paths(self, subject_folder):
        """
        Create output file paths for a subject using the new organized structure.
        
        Args:
            subject_folder (Path): Path to the subject's data folder
            
        Returns:
            dict: Dictionary of output file paths
        """
        subject_name = subject_folder.name
        
        subject_data_dir = self.processed_data_dir / subject_name
        subject_data_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {
            'output_dir': subject_data_dir,
            'corrected_video': subject_data_dir / f"{subject_name}_gaze_corrected_video.mp4",
            'intermediate_csv': subject_data_dir / f"{subject_name}_gaze_output.csv",
            'transformation_history': subject_data_dir / f"{subject_name}_transformation_history.npy",
            'final_csv': subject_data_dir / f"{subject_name}_final_gaze_data.csv",
            'processing_log': self.logs_dir / f"{subject_name}_processing_log.txt",
            'gaze_stats': self.logs_dir / f"{subject_name}_gaze_statistics.json",
            'whole_session_analysis': self.logs_dir / f"{subject_name}_whole_session_analysis.json",
            'heatmap_png': self.figures_dir / f"{subject_name}_heatmap.png",
            'scatter_png': self.figures_dir / f"{subject_name}_scatter.png",
            'contour_png': self.figures_dir / f"{subject_name}_contour.png",
            'dashboard_png': self.figures_dir / f"{subject_name}_dashboard.png",
            # Clinical visualization paths
            'viz_cognitive_fingerprint': self.figures_dir / f"{subject_name}_viz_cognitive_fingerprint.png",
            'viz_main_sequence': self.figures_dir / f"{subject_name}_viz_main_sequence.png",
            'viz_stress_timeline': self.figures_dir / f"{subject_name}_viz_stress_timeline.png",
            'viz_stability_radar': self.figures_dir / f"{subject_name}_viz_stability_radar.png",
        }

        return outputs
    
    def check_existing_outputs(self, output_paths):
        """
        Check if outputs already exist for a subject.

        Args:
            output_paths (dict): Dictionary of output file paths

        Returns:
            bool: True if ALL outputs exist and skip_existing is enabled
        """
        if not self.config['skip_existing']:
            return False

        final_csv = output_paths['final_csv']
        dashboard_png = output_paths['dashboard_png']

        # Check if both CSV and main visualization exist
        if not final_csv.exists():
            return False

        if self.config['generate_heatmaps'] and not dashboard_png.exists():
            return False

        # Check if clinical visualizations exist (if whole-session analysis enabled)
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
        Process a single subject's data with all three steps.
        
        Args:
            subject_folder (Path): Path to the subject's data folder
            
        Returns:
            dict: Processing results for this subject
        """
        subject_name = subject_folder.name
        logger.info(f"Processing subject: {subject_name}")
        logger.info(f"Subject folder: {subject_folder}")
        
        # Create output paths
        output_paths = self.create_output_paths(subject_folder)
        
        # Check if we should skip this subject
        if self.check_existing_outputs(output_paths):
            return {
                'subject_name': subject_name,
                'status': 'skipped',
                'reason': 'Outputs already exist',
                'processing_time': 0
            }
        
        # Initialize result tracking
        subject_start_time = time.time()
        result = {
            'subject_name': subject_name,
            'subject_folder': str(subject_folder),
            'output_paths': {k: str(v) for k, v in output_paths.items()},
            'status': 'failed',
            'error_message': None,
            'processing_time': 0,
            'step1_stats': None,
            'step2_stats': None,
            'step3_stats': None,  # Heatmap analysis stats
            'step4_stats': None   # Whole-session analysis stats
        }
        
        try:
            # Step 1: Process gaze with perspective correction
            transformation_exists = output_paths['transformation_history'].exists()
            if transformation_exists:
                logger.info(f"Step 1: Skipping (transformation history exists)")
                result['step1_stats'] = {'skipped': True, 'reason': 'transformation_history exists'}
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
                logger.info(f"Step 1 completed: {step1_stats['frames_with_valid_homography']} frames with valid homography")

            # Step 2: Create final high-resolution CSV
            final_csv_exists = output_paths['final_csv'].exists()
            if final_csv_exists:
                logger.info(f"Step 2: Skipping (final CSV exists)")
                result['step2_stats'] = {'skipped': True, 'success': True, 'reason': 'final_csv exists'}
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
                    logger.error(f"Step 2 failed: Could not create final CSV")
                    return result

                result['step2_stats'] = step2_stats
                logger.info(f"Step 2 completed: {step2_stats['valid_transformations']} valid transformations ({step2_stats['valid_percentage']:.1f}%)")
            
            # Step 3: Generate heatmap visualizations
            if self.config['generate_heatmaps'] and self.heatmap_analyzer:
                logger.info(f"Step 3: Generating gaze heatmap visualizations...")
                
                # Configure the heatmap analyzer to use our organized directories
                step3_stats = self.heatmap_analyzer.analyze_subject(
                    csv_path=str(output_paths['final_csv']),
                    output_dir=str(self.figures_dir),  # Send images to figures directory
                    subject_name=subject_name
                )
                
                # Also save statistics to logs directory
                if step3_stats.get('success', False) and step3_stats.get('statistics'):
                    stats_file = output_paths['gaze_stats']
                    try:
                        with open(stats_file, 'w') as f:
                            json.dump(step3_stats['statistics'], f, indent=2, default=str)
                        logger.info(f"Gaze statistics saved to: {stats_file}")
                    except Exception as e:
                        logger.warning(f"Warning: Could not save gaze statistics: {e}")
                
                if step3_stats.get('success', False):
                    result['step3_stats'] = step3_stats
                    num_visualizations = len(step3_stats.get('visualizations_created', []))
                    valid_gaze_count = step3_stats.get('statistics', {}).get('filtered_samples', 0)
                    logger.info(f"Step 3 completed: {num_visualizations} visualizations created ({valid_gaze_count:,} gaze points)")
                else:
                    logger.warning(f"Step 3 warning: {step3_stats.get('error', 'Could not create visualizations')}")
                    # Don't fail the entire process if only visualizations fail
                    result['step3_stats'] = step3_stats

            # Step 4: Whole-Session Analysis (Physics-based metrics)
            analyzer = None  # Initialize for potential use in Step 5
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
                    analysis_file = output_paths['whole_session_analysis']
                    with open(analysis_file, 'w') as f:
                        # Convert any numpy types to Python types for JSON
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
                    logger.info(f"Step 4 completed: Whole-session analysis saved")
                    logger.info(f"  Recording duration: {step4_stats.get('recording_duration_s', 0):.1f} s")

                    # Log key metrics
                    goal_a = step4_stats.get('goal_a_oculometric_efficiency', {})
                    goal_b = step4_stats.get('goal_b_cognitive_load', {})
                    goal_c = step4_stats.get('goal_c_motor_stability', {})

                    if goal_a:
                        logger.info(f"  Fixation rate: {goal_a.get('fixation_rate_hz', 0):.2f} Hz")
                    if goal_b:
                        logger.info(f"  Pupil residual: {goal_b.get('mean_residual', 0):.4f} mm")
                    if goal_c:
                        logger.info(f"  Total head rotation: {goal_c.get('total_rotation_deg', 0):.1f} deg")

                except Exception as e:
                    logger.warning(f"Step 4 warning: Could not complete whole-session analysis: {e}")
                    result['step4_stats'] = {'error': str(e)}
                    analyzer = None  # Ensure analyzer is None if step 4 failed

            # Step 5: Generate clinical visualizations
            if self.config.get('run_whole_session_analysis', True) and analyzer is not None:
                logger.info(f"Step 5: Generating clinical visualizations...")

                try:
                    step5_stats = self.gaze_visualizer.create_all_visualizations(
                        analyzer=analyzer,
                        output_dir=str(self.figures_dir),
                        subject_name=subject_name
                    )

                    result['step5_stats'] = step5_stats
                    num_viz = len(step5_stats.get('created', {}))
                    logger.info(f"Step 5 completed: {num_viz} visualizations created")

                    if step5_stats.get('errors'):
                        for err in step5_stats['errors']:
                            logger.warning(f"  Visualization error: {err}")

                except Exception as e:
                    logger.warning(f"Step 5 warning: Could not create visualizations: {e}")
                    result['step5_stats'] = {'error': str(e)}

            # Mark as successful if we got through at least steps 1 and 2
            result['status'] = 'success'
        
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"Processing failed: {e}")
            logger.exception("Traceback:")
        
        # Calculate processing time
        result['processing_time'] = time.time() - subject_start_time
        
        # Save processing log
        save_processing_log(result, output_paths['processing_log'])
        
        return result
    
    def run(self):
        """
        Run the enhanced batch processing for all discovered subjects.
        
        Returns:
            dict: Overall processing results
        """
        self.start_time = datetime.now()
        logger.info("Starting enhanced batch processing with heatmap generation and skip list...")
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.config['generate_heatmaps']:
            logger.info("Heatmap generation: ENABLED")
        else:
            logger.info("Heatmap generation: DISABLED")
        
        # Show skip list information
        skip_list = self.config.get('subjects_to_skip', [])
        if skip_list:
            logger.info(f"Skip list: ENABLED ({len(skip_list)} subjects)")
        else:
            logger.info("Skip list: DISABLED")
        
        # Discover subject folders
        subject_folders, self.skipped_subjects = discover_subject_folders(
            self.config['input_base_dir'],
            self.config['video_filename'],
            self.config['gaze_filename'],
            self.config['subject_folder_pattern'],
            self.config['subjects_to_skip']
        )
        
        if not subject_folders:
            logger.warning("No valid subject folders found. Exiting.")
            return {'success': False, 'error': 'No valid subject folders found'}
        
        self.total_subjects = len(subject_folders)
        
        # Process each subject
        for i, subject_folder in enumerate(subject_folders, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing subject {i}/{self.total_subjects}")
            logger.info(f"{'='*70}")
            
            result = self.process_single_subject(subject_folder)
            self.results.append(result)

            # Update counters
            if result['status'] == 'success':
                self.successful_subjects += 1
            elif result['status'] == 'failed':
                self.failed_subjects += 1

            # Free memory between subjects
            gc.collect()

        # Create summary report
        create_summary_report(
            self.config, self.results, self.skipped_subjects, self.total_subjects,
            self.successful_subjects, self.failed_subjects, self.start_time,
            self.logs_dir, self.figures_dir, self.processed_data_dir
        )
        
        # Calculate heatmap statistics
        heatmap_successes = sum(1 for r in self.results
                               if r.get('step3_stats', {}).get('success', False))

        # Calculate whole-session analysis statistics
        ws_successes = sum(1 for r in self.results
                          if r.get('step4_stats') and not r.get('step4_stats', {}).get('error'))

        # Calculate clinical visualization statistics
        viz_successes = sum(1 for r in self.results
                           if r.get('step5_stats', {}).get('success', False))

        # Print final summary
        logger.info(f"ENHANCED BATCH PROCESSING COMPLETE!")
        logger.info(f"{'='*70}")
        logger.info(f"Total subjects discovered: {self.total_subjects + self.skipped_subjects}")
        logger.info(f"Subjects processed: {self.total_subjects}")
        logger.info(f"Successful: {self.successful_subjects}")
        logger.info(f"Failed: {self.failed_subjects}")
        if self.skipped_subjects > 0:
            logger.info(f"Skipped (skip list): {self.skipped_subjects}")
        skipped_existing = self.total_subjects - self.successful_subjects - self.failed_subjects
        if skipped_existing > 0:
            logger.info(f"Skipped (existing outputs): {skipped_existing}")
        logger.info(f"Success rate: {(self.successful_subjects / self.total_subjects * 100) if self.total_subjects > 0 else 0:.1f}%")

        if self.config['generate_heatmaps']:
            logger.info(f"Heatmaps created: {heatmap_successes}/{self.total_subjects} ({(heatmap_successes / self.total_subjects * 100) if self.total_subjects > 0 else 0:.1f}%)")

        if self.config.get('run_whole_session_analysis', True):
            logger.info(f"Whole-session analysis: {ws_successes}/{self.total_subjects} ({(ws_successes / self.total_subjects * 100) if self.total_subjects > 0 else 0:.1f}%)")
            logger.info(f"Clinical visualizations: {viz_successes}/{self.total_subjects} ({(viz_successes / self.total_subjects * 100) if self.total_subjects > 0 else 0:.1f}%)")

        logger.info(f"Total time: {time.time() - self.start_time.timestamp():.1f} seconds")
        logger.info(f"Results organized in: {self.output_root}")
        logger.info(f"  - Images: {self.figures_dir}")
        logger.info(f"  - Logs: {self.logs_dir}")
        logger.info(f"  - Processed Data: {self.processed_data_dir}")
        
        return {
            'success': True,
            'total_subjects': self.total_subjects,
            'successful_subjects': self.successful_subjects,
            'failed_subjects': self.failed_subjects,
            'skipped_subjects_skip_list': self.skipped_subjects,
            'heatmap_successes': heatmap_successes,
            'whole_session_successes': ws_successes,
            'visualization_successes': viz_successes,
            'results': self.results
        }


def batch_process_subjects(config=None):
    """
    Convenience function for enhanced batch processing subjects.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Processing results
    """
    processor = EnhancedBatchProcessor(config)
    return processor.run()


def main():
    """
    Main function for command line execution.
    """
    try:
        # --- Find project paths automatically ---
        script_path = Path(__file__).resolve()
        src_dir = script_path.parent
        project_root = src_dir.parent

        logger.info(f"Script location: {script_path}")
        logger.info(f"Source directory: {src_dir}")
        logger.info(f"Project root: {project_root}")

        # Load configuration from file
        config_path = project_root / 'config.json'
        if not config_path.exists():
            logger.warning(f"Configuration file not found at {config_path}. Creating a default one.")
            default_config_data = {
                "input_base_dir": "data/raw",
                "subjects_to_skip": [],
                "subject_folder_pattern": "*",
                "skip_existing": True,
                "generate_heatmaps": True,
                "create_summary_report": True,
                "heatmap_config": {
                    "figure_size": [12, 8],
                    "dpi": 300,
                    "color_scheme": "viridis"
                },
                "processing_options": {
                    "show_video": False
                }
            }
            with open(config_path, 'w') as f:
                json.dump(default_config_data, f, indent=4)
            logger.info(f"Default config.json created at {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Set the input directory dynamically
        input_dir = project_root / config.get('input_base_dir', 'data/raw')
        config['input_base_dir'] = input_dir

        # Verify the input directory exists
        if not input_dir.exists():
            logger.error(f"ERROR: Input directory does not exist: {input_dir}")
            logger.error(f"Please ensure your raw data is in: {input_dir}")
            return False
        
        logger.info(f"Input directory confirmed: {input_dir}")

        logger.info(f"Configuration loaded from: {config_path}")
        logger.info(f"  Input (raw data): {config['input_base_dir']}")
        logger.info(f"  Output (results):")
        logger.info(f"    - Processed data: {project_root}/data/processed/")
        logger.info(f"    - Figures: {project_root}/reports/figures/")
        logger.info(f"    - Logs: {project_root}/reports/logs/")

        # Show skip list information
        skip_list = config.get('subjects_to_skip', [])
        if skip_list:
            logger.info(f"  Skip list: {len(skip_list)} subjects will be skipped: {skip_list}")
        else:
            logger.info(f"  Skip list: No subjects to skip (empty list)")

        results = batch_process_subjects(config)
        
        if results and results.get('success'):
            logger.info(f"Enhanced batch processing completed successfully!")
            return True
        else:
            logger.warning(f"Enhanced batch processing failed or had issues.")
            return False
            
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}")
        logger.exception("Traceback:")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
