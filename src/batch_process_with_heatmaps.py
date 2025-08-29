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
from pathlib import Path
from datetime import datetime
import json
import traceback

# --- Add subdirectories to the Python path ---
# This allows the script to find the refactored modules
# This gets the directory the script is in (the 'src' directory)
src_dir = Path(__file__).resolve().parent
# Add the 'processing' and 'analysis' subfolders to Python's search path
sys.path.append(str(src_dir / 'processing'))
sys.path.append(str(src_dir / 'analysis'))

# Import the refactored processing functions
try:
    from gaze_on_perspective_corrected_frames_refactored import process_gaze_with_perspective_correction
    from create_final_csv_refactored import create_final_gaze_csv
    from gaze_heatmap_analysis import GazeHeatmapAnalyzer  # NEW!
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure all required scripts are in the same directory:")
    print("  - gaze_on_perspective_corrected_frames_refactored.py")
    print("  - create_final_csv_refactored.py")
    print("  - gaze_heatmap_analysis.py")
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

        # Default configuration
        self.config = {
            'input_base_dir': self.project_root / 'data' / 'raw',
            'output_base_dir': self.project_root,
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
            
            # NEW: Heatmap analysis configuration
            'generate_heatmaps': True,  # Enable/disable heatmap generation
            'heatmap_config': {
                'figure_size': (12, 8),
                'dpi': 300,
                'color_scheme': 'viridis',  # viridis, plasma, inferno, magma, hot
                'heatmap_bins': 50,
                'gaussian_sigma': 1.0,
                'output_format': 'png',
                'create_heatmap': True,
                'create_scatter': True,
                'create_contour': True,
                'create_combined': True,  # Dashboard with all visualizations
                'show_stats_overlay': True,
                'save_stats': True,
                'min_valid_points': 100
            }
        }
        
        # Update with user config if provided
        if config:
            self._update_config(config)
        
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
        
        print(f"Output directory structure created:")
        print(f"  Root: {self.output_root}")
        print(f"  Processed Data: {self.processed_data_dir}")
        print(f"  Figures: {self.figures_dir}")
        print(f"  Logs: {self.logs_dir}")
    
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
    
    def discover_subject_folders(self):
        """
        Discover subject folders in the input base directory, excluding those in the skip list.
        
        Returns:
            list: List of valid subject folder paths (excluding skipped subjects)
        """
        input_path = Path(self.config['input_base_dir'])
        
        if not input_path.exists():
            print(f"Input directory does not exist: {input_path}")
            return []
        
        print(f"Scanning for subject folders in: {input_path}")
        
        # Get the skip list and convert to set for faster lookup
        subjects_to_skip = set(self.config.get('subjects_to_skip', []))
        if subjects_to_skip:
            print(f"Skip list contains {len(subjects_to_skip)} subjects: {sorted(subjects_to_skip)}")
        
        pattern = self.config['subject_folder_pattern']
        subject_folders = []
        skipped_count = 0
        
        for folder_path in input_path.glob(pattern):
            if folder_path.is_dir():
                folder_name = folder_path.name
                
                if folder_name in subjects_to_skip:
                    print(f"Skipping {folder_name}: Subject is in skip list")
                    skipped_count += 1
                    continue
                
                video_file = folder_path / self.config['video_filename']
                gaze_file = folder_path / self.config['gaze_filename']
                
                if video_file.exists() and gaze_file.exists():
                    subject_folders.append(folder_path)
                    print(f"Found valid subject folder: {folder_path.name}")
                else:
                    missing_files = []
                    if not video_file.exists():
                        missing_files.append(self.config['video_filename'])
                    if not gaze_file.exists():
                        missing_files.append(self.config['gaze_filename'])
                    print(f"Skipping {folder_path.name}: Missing {', '.join(missing_files)}")
        
        # Update the skipped count for reporting
        self.skipped_subjects = skipped_count
        
        print(f"Found {len(subject_folders)} valid subject folders")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} subjects due to skip list")
        
        return sorted(subject_folders)
    
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
            'heatmap_png': self.figures_dir / f"{subject_name}_heatmap.png",
            'scatter_png': self.figures_dir / f"{subject_name}_scatter.png",
            'contour_png': self.figures_dir / f"{subject_name}_contour.png",
            'dashboard_png': self.figures_dir / f"{subject_name}_dashboard.png",
        }
        
        return outputs
    
    def check_existing_outputs(self, output_paths):
        """
        Check if outputs already exist for a subject.
        
        Args:
            output_paths (dict): Dictionary of output file paths
            
        Returns:
            bool: True if final output exists and skip_existing is enabled
        """
        if not self.config['skip_existing']:
            return False
        
        final_csv = output_paths['final_csv']
        dashboard_png = output_paths['dashboard_png']
        
        # Check if both CSV and main visualization exist
        if final_csv.exists():
            if not self.config['generate_heatmaps'] or dashboard_png.exists():
                print(f"Skipping: Outputs already exist")
                return True
        
        return False
    
    def process_single_subject(self, subject_folder):
        """
        Process a single subject's data with all three steps.
        
        Args:
            subject_folder (Path): Path to the subject's data folder
            
        Returns:
            dict: Processing results for this subject
        """
        subject_name = subject_folder.name
        print(f"\nProcessing subject: {subject_name}")
        print(f"Subject folder: {subject_folder}")
        
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
            'step3_stats': None  # NEW: Heatmap analysis stats
        }
        
        try:
            # Step 1: Process gaze with perspective correction
            print(f"Step 1: Processing video with gaze data...")
            
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
            print(f"Step 1 completed: {step1_stats['frames_with_valid_homography']} frames with valid homography")
            
            # Step 2: Create final high-resolution CSV
            print(f"Step 2: Creating final high-resolution gaze CSV...")
            
            step2_stats = create_final_gaze_csv(
                gaze_file_path=str(subject_folder / self.config['gaze_filename']),
                transformation_history_path=str(output_paths['transformation_history']),
                output_csv_path=str(output_paths['final_csv']),
                frame_width=self.config['frame_width'],
                frame_height=self.config['frame_height']
            )
            
            if not step2_stats or not step2_stats.get('success', False):
                result['error_message'] = "Step 2 failed: Could not create final CSV"
                print(f"Step 2 failed: Could not create final CSV")
                return result
            
            result['step2_stats'] = step2_stats
            print(f"Step 2 completed: {step2_stats['valid_transformations']} valid transformations ({step2_stats['valid_percentage']:.1f}%)")
            
            # Step 3: Generate heatmap visualizations
            if self.config['generate_heatmaps'] and self.heatmap_analyzer:
                print(f"Step 3: Generating gaze heatmap visualizations...")
                
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
                        print(f"Gaze statistics saved to: {stats_file}")
                    except Exception as e:
                        print(f"Warning: Could not save gaze statistics: {e}")
                
                if step3_stats.get('success', False):
                    result['step3_stats'] = step3_stats
                    num_visualizations = len(step3_stats.get('visualizations_created', []))
                    valid_gaze_count = step3_stats.get('statistics', {}).get('filtered_samples', 0)
                    print(f"Step 3 completed: {num_visualizations} visualizations created ({valid_gaze_count:,} gaze points)")
                else:
                    print(f"Step 3 warning: {step3_stats.get('error', 'Could not create visualizations')}")
                    # Don't fail the entire process if only visualizations fail
                    result['step3_stats'] = step3_stats
            
            # Mark as successful if we got through at least steps 1 and 2
            result['status'] = 'success'
        
        except Exception as e:
            result['error_message'] = str(e)
            print(f"Processing failed: {e}")
            traceback.print_exc()
        
        # Calculate processing time
        result['processing_time'] = time.time() - subject_start_time
        
        # Save processing log
        self.save_processing_log(result, output_paths['processing_log'])
        
        return result
    
    def save_processing_log(self, result, log_path):
        """
        Save enhanced processing log for a subject.
        
        Args:
            result (dict): Processing result dictionary
            log_path (Path): Path to save the log file
        """
        try:
            with open(log_path, 'w') as f:
                f.write(f"Enhanced Processing Log for {result['subject_name']}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Processing time: {result['processing_time']:.2f} seconds\n\n")
                
                if result['error_message']:
                    f.write(f"Error: {result['error_message']}\n\n")
                
                if result['step1_stats']:
                    f.write("Step 1 - Video Processing Stats:\n")
                    for key, value in result['step1_stats'].items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                if result['step2_stats']:
                    f.write("Step 2 - Final CSV Stats:\n")
                    for key, value in result['step2_stats'].items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # Step 3 stats
                if result['step3_stats']:
                    f.write("Step 3 - Heatmap Analysis Stats:\n")
                    stats = result['step3_stats']
                    if stats.get('success'):
                        f.write(f"  Visualizations created: {len(stats.get('visualizations_created', []))}\n")
                        if 'statistics' in stats:
                            gaze_stats = stats['statistics']
                            f.write(f"  Valid gaze points: {gaze_stats.get('filtered_samples', 0):,}\n")
                            f.write(f"  Data quality: {gaze_stats.get('filtered_percentage', 0):.1f}%\n")
                    else:
                        f.write(f"  Error: {stats.get('error', 'Unknown error')}\n")
                    f.write("\n")
                
                f.write("Output Files:\n")
                for key, path in result['output_paths'].items():
                    exists = "Yes" if Path(path).exists() else "No"
                    f.write(f"  {key}: {exists} {path}\n")
        
        except Exception as e:
            print(f"Could not save processing log: {e}")
    
    def create_summary_report(self):
        """
        Create an enhanced summary report of all processed subjects with timestamps and skip list info.
        """
        if not self.config['create_summary_report']:
            return
        
        print(f"\nCreating enhanced summary report...")
        
        # Create timestamped filenames
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        summary_json_path = self.logs_dir / f"batch_summary_{timestamp}.json"
        summary_txt_path = self.logs_dir / f"batch_summary_{timestamp}.txt"
        
        # Calculate additional statistics
        heatmap_successes = sum(1 for r in self.results
                               if r.get('step3_stats', {}).get('success', False))
        
        total_gaze_points = sum(r.get('step3_stats', {}).get('statistics', {}).get('filtered_samples', 0)
                               for r in self.results if r.get('step3_stats'))
        
        # Count different types of skipped subjects
        output_exists_skipped = sum(1 for r in self.results
                                   if r['status'] == 'skipped' and r.get('reason') == 'Outputs already exist')
        
        summary = {
            'processing_session': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'total_duration': time.time() - self.start_time.timestamp() if self.start_time else 0,
                'config': self.config
            },
            'overall_stats': {
                'total_subjects': self.total_subjects,
                'successful_subjects': self.successful_subjects,
                'failed_subjects': self.failed_subjects,
                'skipped_subjects_skip_list': self.skipped_subjects,  # NEW: From skip list
                'skipped_subjects_output_exists': output_exists_skipped,  # From existing outputs
                'total_skipped_subjects': self.skipped_subjects + output_exists_skipped,
                'success_rate': (self.successful_subjects / self.total_subjects * 100) if self.total_subjects > 0 else 0,
                
                # Skip list information
                'skip_list_enabled': len(self.config.get('subjects_to_skip', [])) > 0,
                'skip_list_count': len(self.config.get('subjects_to_skip', [])),
                'skip_list_subjects': self.config.get('subjects_to_skip', []),
                
                # Heatmap-specific statistics
                'heatmap_generation_enabled': self.config['generate_heatmaps'],
                'heatmap_successes': heatmap_successes,
                'heatmap_success_rate': (heatmap_successes / self.total_subjects * 100) if self.total_subjects > 0 else 0,
                'total_gaze_points_analyzed': total_gaze_points
            },
            'subject_results': self.results
        }
        
        try:
            # Save JSON summary
            with open(summary_json_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"Enhanced summary report saved to: {summary_json_path}")
            
            # Create enhanced text summary
            with open(summary_txt_path, 'w') as f:
                f.write("ENHANCED BATCH PROCESSING SUMMARY\n")
                f.write("="*60 + "\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write(f"Total subjects discovered: {self.total_subjects + self.skipped_subjects}\n")
                f.write(f"Subjects processed: {self.total_subjects}\n")
                f.write(f"Successful (all steps): {self.successful_subjects}\n")
                f.write(f"Failed: {self.failed_subjects}\n")
                f.write(f"Skipped (skip list): {self.skipped_subjects}\n")
                f.write(f"Skipped (existing outputs): {output_exists_skipped}\n")
                f.write(f"Overall success rate: {summary['overall_stats']['success_rate']:.1f}%\n\n")
                
                # Skip list information
                if self.skipped_subjects > 0:
                    f.write("SKIP LIST INFORMATION:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Skip list enabled: Yes\n")
                    f.write(f"Subjects in skip list: {len(self.config.get('subjects_to_skip', []))}\n")
                    skip_list = self.config.get('subjects_to_skip', [])
                    if skip_list:
                        f.write(f"Skip list contents: {', '.join(sorted(skip_list))}\n")
                    f.write(f"Subjects actually skipped: {self.skipped_subjects}\n\n")
                
                if self.config['generate_heatmaps']:
                    f.write(f"Heatmap visualizations created: {heatmap_successes}/{self.total_subjects}\n")
                    f.write(f"Heatmap success rate: {summary['overall_stats']['heatmap_success_rate']:.1f}%\n")
                    f.write(f"Total gaze points analyzed: {total_gaze_points:,}\n\n")
                
                f.write("OUTPUT DIRECTORY STRUCTURE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"All images (.png): {self.figures_dir}\n")
                f.write(f"All logs (.txt, .json): {self.logs_dir}\n")
                f.write(f"All processed data: {self.processed_data_dir}\n\n")
                
                f.write("INDIVIDUAL RESULTS:\n")
                f.write("-" * 40 + "\n")
                for result in self.results:
                    # More readable status icon assignment
                    if result['status'] == 'success':
                        status_icon = "✓"
                    elif result['status'] == 'failed':
                        status_icon = "✗"
                    else:  # skipped or other status
                        status_icon = "~"
                    
                    f.write(f"{status_icon} {result['subject_name']}: {result['status']}")
                    
                    if result['status'] == 'success':
                        if result.get('step2_stats'):
                            f.write(f" ({result['step2_stats']['valid_percentage']:.1f}% valid gaze)")
                        
                        if result.get('step3_stats', {}).get('success'):
                            viz_count = len(result['step3_stats'].get('visualizations_created', []))
                            f.write(f", {viz_count} visualizations")
                    
                    elif result['status'] == 'failed':
                        f.write(f" - {result.get('error_message', 'Unknown error')}")
                    
                    f.write(f" ({result['processing_time']:.1f}s)\n")
            
            print(f"Enhanced text summary saved to: {summary_txt_path}")
        
        except Exception as e:
            print(f"Could not save summary report: {e}")
    
    def run(self):
        """
        Run the enhanced batch processing for all discovered subjects.
        
        Returns:
            dict: Overall processing results
        """
        self.start_time = datetime.now()
        print("Starting enhanced batch processing with heatmap generation and skip list...")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.config['generate_heatmaps']:
            print("Heatmap generation: ENABLED")
        else:
            print("Heatmap generation: DISABLED")
        
        # Show skip list information
        skip_list = self.config.get('subjects_to_skip', [])
        if skip_list:
            print(f"Skip list: ENABLED ({len(skip_list)} subjects)")
        else:
            print("Skip list: DISABLED")
        
        # Discover subject folders
        subject_folders = self.discover_subject_folders()
        
        if not subject_folders:
            print("No valid subject folders found. Exiting.")
            return {'success': False, 'error': 'No valid subject folders found'}
        
        self.total_subjects = len(subject_folders)
        
        # Process each subject
        for i, subject_folder in enumerate(subject_folders, 1):
            print(f"\n{'='*70}")
            print(f"Processing subject {i}/{self.total_subjects}")
            print(f"{'='*70}")
            
            result = self.process_single_subject(subject_folder)
            self.results.append(result)
            
            # Update counters
            if result['status'] == 'success':
                self.successful_subjects += 1
            elif result['status'] == 'failed':
                self.failed_subjects += 1
        
        # Create summary report
        self.create_summary_report()
        
        # Calculate heatmap statistics
        heatmap_successes = sum(1 for r in self.results
                               if r.get('step3_stats', {}).get('success', False))
        
        # Print final summary
        print(f"\nENHANCED BATCH PROCESSING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total subjects discovered: {self.total_subjects + self.skipped_subjects}")
        print(f"Subjects processed: {self.total_subjects}")
        print(f"Successful: {self.successful_subjects}")
        print(f"Failed: {self.failed_subjects}")
        if self.skipped_subjects > 0:
            print(f"Skipped (skip list): {self.skipped_subjects}")
        skipped_existing = self.total_subjects - self.successful_subjects - self.failed_subjects
        if skipped_existing > 0:
            print(f"Skipped (existing outputs): {skipped_existing}")
        print(f"Success rate: {(self.successful_subjects / self.total_subjects * 100) if self.total_subjects > 0 else 0:.1f}%")
        
        if self.config['generate_heatmaps']:
            print(f"Heatmaps created: {heatmap_successes}/{self.total_subjects} ({(heatmap_successes / self.total_subjects * 100) if self.total_subjects > 0 else 0:.1f}%)")
        
        print(f"Total time: {time.time() - self.start_time.timestamp():.1f} seconds")
        print(f"Results organized in: {self.output_root}")
        print(f"  - Images: {self.figures_dir}")
        print(f"  - Logs: {self.logs_dir}")
        print(f"  - Processed Data: {self.processed_data_dir}")
        
        return {
            'success': True,
            'total_subjects': self.total_subjects,
            'successful_subjects': self.successful_subjects,
            'failed_subjects': self.failed_subjects,
            'skipped_subjects_skip_list': self.skipped_subjects,
            'heatmap_successes': heatmap_successes,
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
    # --- Find project paths automatically ---
    script_path = Path(__file__).resolve()
    src_dir = script_path.parent
    project_root = src_dir.parent

    print(f"Script location: {script_path}")
    print(f"Source directory: {src_dir}")
    print(f"Project root: {project_root}")
    
    input_dir = project_root / 'data' / 'raw'
    
    # Verify the input directory exists
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        print(f"Please ensure your raw data is in: {input_dir}")
        return False
    
    print(f"Input directory confirmed: {input_dir}")

    # --- ENHANCED Configuration with Skip List ---
    config = {
        # define base directory
        'input_base_dir': input_dir,
        
        # NEW: Skip List - Add known problematic subjects here
        'subjects_to_skip': [
            '20231012T122519Z',
        ],
        
        # Processing settings
        'subject_folder_pattern': '*',  # Match any folder
        'skip_existing': True,
        'generate_heatmaps': True,
        'create_summary_report': True,
        'heatmap_config': {
            'figure_size': (12, 8),
            'dpi': 300,
            'color_scheme': 'viridis'
        },
        'processing_options': {
            'show_video': False
        }
    }

    print(f"\nConfiguration:")
    print(f"  Input (raw data): {config['input_base_dir']}")
    print(f"  Output (results):")
    print(f"    - Processed data: {project_root}/data/processed/")
    print(f"    - Figures: {project_root}/reports/figures/")
    print(f"    - Logs: {project_root}/reports/logs/")
    
    # Show skip list information
    skip_list = config.get('subjects_to_skip', [])
    if skip_list:
        print(f"  Skip list: {len(skip_list)} subjects will be skipped: {skip_list}")
    else:
        print(f"  Skip list: No subjects to skip (empty list)")

    try:
        results = batch_process_subjects(config)
        
        if results and results.get('success'):
            print(f"\nEnhanced batch processing completed successfully!")
            return True
        else:
            print(f"\nEnhanced batch processing failed or had issues.")
            return False
            
    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
