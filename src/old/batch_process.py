"""
batch_process.py - Master script for batch processing multiple subjects' gaze data

"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json
import traceback

# Import the refactored processing functions
try:
    from gaze_on_perspective_corrected_frames_refactored import process_gaze_with_perspective_correction
    from create_final_csv_refactored import create_final_gaze_csv
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure both refactored scripts are in the same directory as this script.")
    sys.exit(1)


class BatchProcessor:
    """
    Batch processor class for handling multiple subjects' gaze data processing.
    """
    
    def __init__(self, config=None):
        """
        Initialize the batch processor with configuration.
        
        Args:
            config (dict): Configuration dictionary with processing parameters
        """
        # Default configuration
        self.config = {
            'input_base_dir': '/Users/sachitanand/Lab_PatrickMayo/Projects/Surgical_OpenCV',
            'output_base_dir': './batch_processing_results',
            'video_filename': 'scenevideo.mp4',
            'gaze_filename': 'gazedata.gz',
            'subject_folder_pattern': '*',  # Pattern to match subject folders
            'output_width': 1000,
            'target_markers': [13, 14, 15, 16],
            'frame_width': 1920,
            'frame_height': 1080,
            'processing_options': {
                'use_preselected_parameters': False,
                'use_frame_preprocessing': False,
                'use_outer_points': False,
                'show_video': False  # Always False for batch processing
            },
            'skip_existing': True,  # Skip subjects that already have outputs
            'create_summary_report': True
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        # Initialize tracking variables
        self.results = []
        self.start_time = None
        self.total_subjects = 0
        self.successful_subjects = 0
        self.failed_subjects = 0
    
    def discover_subject_folders(self):
        """
        Discover subject folders in the input base directory.
        
        Returns:
            list: List of valid subject folder paths
        """
        input_path = Path(self.config['input_base_dir'])
        
        if not input_path.exists():
            print(f"Input directory does not exist: {input_path}")
            return []
        
        print(f"Scanning for subject folders in: {input_path}")
        
        # Find all directories matching the pattern
        pattern = self.config['subject_folder_pattern']
        subject_folders = []
        
        for folder_path in input_path.glob(pattern):
            if folder_path.is_dir():
                # Check if folder contains required files
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
        
        print(f"Found {len(subject_folders)} valid subject folders")
        return sorted(subject_folders)
    
    def create_output_paths(self, subject_folder):
        """
        Create output file paths for a subject.
        
        Args:
            subject_folder (Path): Path to the subject's data folder
            
        Returns:
            dict: Dictionary of output file paths
        """
        subject_name = subject_folder.name
        output_dir = Path(self.config['output_base_dir']) / subject_name
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {
            'output_dir': output_dir,
            'corrected_video': output_dir / f"{subject_name}_gaze_corrected_video.mp4",
            'intermediate_csv': output_dir / f"{subject_name}_gaze_output.csv",
            'transformation_history': output_dir / f"{subject_name}_transformation_history.npy",
            'final_csv': output_dir / f"{subject_name}_final_gaze_data.csv",
            'processing_log': output_dir / f"{subject_name}_processing_log.txt"
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
        if final_csv.exists():
            print(f"Skipping: Final CSV already exists at {final_csv}")
            return True
        
        return False
    
    def process_single_subject(self, subject_folder):
        """
        Process a single subject's data.
        
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
                'reason': 'Final CSV already exists',
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
            'step2_stats': None
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
            
            if step2_stats and step2_stats.get('success', False):
                result['step2_stats'] = step2_stats
                result['status'] = 'success'
                print(f"Step 2 completed: {step2_stats['valid_transformations']} valid transformations ({step2_stats['valid_percentage']:.1f}%)")
            else:
                result['error_message'] = "Step 2 failed: Could not create final CSV"
                print(f"Step 2 failed: Could not create final CSV")
        
        except Exception as e:
            result['error_message'] = str(e)
            print(f"Processing failed: {e}")
            # Print full traceback for debugging
            print(f"Error details:")
            traceback.print_exc()
        
        # Calculate processing time
        result['processing_time'] = time.time() - subject_start_time
        
        # Save processing log
        self.save_processing_log(result, output_paths['processing_log'])
        
        return result
    
    def save_processing_log(self, result, log_path):
        """
        Save processing log for a subject.
        
        Args:
            result (dict): Processing result dictionary
            log_path (Path): Path to save the log file
        """
        try:
            with open(log_path, 'w') as f:
                f.write(f"Processing Log for {result['subject_name']}\n")
                f.write(f"{'='*50}\n")
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
                
                f.write("Output Files:\n")
                for key, path in result['output_paths'].items():
                    exists = "Yes" if Path(path).exists() else "No"
                    f.write(f"  {key}: {exists} {path}\n")
        
        except Exception as e:
            print(f"Could not save processing log: {e}")
    
    def create_summary_report(self):
        """
        Create a summary report of all processed subjects.
        """
        if not self.config['create_summary_report']:
            return
        
        print(f"\nCreating summary report...")
        
        summary_path = Path(self.config['output_base_dir']) / "batch_processing_summary.json"
        
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
                'skipped_subjects': self.total_subjects - self.successful_subjects - self.failed_subjects,
                'success_rate': (self.successful_subjects / self.total_subjects * 100) if self.total_subjects > 0 else 0
            },
            'subject_results': self.results
        }
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"Summary report saved to: {summary_path}")
            
            # Also create a human-readable text summary
            text_summary_path = Path(self.config['output_base_dir']) / "batch_processing_summary.txt"
            with open(text_summary_path, 'w') as f:
                f.write("BATCH PROCESSING SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Processed: {self.total_subjects} subjects\n")
                f.write(f"Successful: {self.successful_subjects}\n")
                f.write(f"Failed: {self.failed_subjects}\n")
                f.write(f"Skipped: {self.total_subjects - self.successful_subjects - self.failed_subjects}\n")
                f.write(f"Success Rate: {summary['overall_stats']['success_rate']:.1f}%\n\n")
                
                f.write("INDIVIDUAL RESULTS:\n")
                f.write("-" * 30 + "\n")
                for result in self.results:
                    f.write(f"{result['subject_name']}: {result['status']}")
                    if result['status'] == 'success' and result.get('step2_stats'):
                        f.write(f" ({result['step2_stats']['valid_percentage']:.1f}% valid gaze)")
                    elif result['status'] == 'failed':
                        f.write(f" - {result.get('error_message', 'Unknown error')}")
                    f.write(f" ({result['processing_time']:.1f}s)\n")
            
            print(f"Text summary saved to: {text_summary_path}")
        
        except Exception as e:
            print(f"Could not save summary report: {e}")
    
    def run(self):
        """
        Run the batch processing for all discovered subjects.
        
        Returns:
            dict: Overall processing results
        """
        self.start_time = datetime.now()
        print("Starting batch processing...")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create output base directory
        Path(self.config['output_base_dir']).mkdir(parents=True, exist_ok=True)
        
        # Discover subject folders
        subject_folders = self.discover_subject_folders()
        
        if not subject_folders:
            print("No valid subject folders found. Exiting.")
            return {'success': False, 'error': 'No valid subject folders found'}
        
        self.total_subjects = len(subject_folders)
        
        # Process each subject
        for i, subject_folder in enumerate(subject_folders, 1):
            print(f"\n{'='*60}")
            print(f"Processing subject {i}/{self.total_subjects}")
            print(f"{'='*60}")
            
            result = self.process_single_subject(subject_folder)
            self.results.append(result)
            
            # Update counters
            if result['status'] == 'success':
                self.successful_subjects += 1
            elif result['status'] == 'failed':
                self.failed_subjects += 1
        
        # Create summary report
        self.create_summary_report()
        
        # Print final summary
        print(f"\nBATCH PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"Total subjects: {self.total_subjects}")
        print(f"Successful: {self.successful_subjects}")
        print(f"Failed: {self.failed_subjects}")
        print(f"Skipped: {self.total_subjects - self.successful_subjects - self.failed_subjects}")
        print(f"Success rate: {(self.successful_subjects / self.total_subjects * 100) if self.total_subjects > 0 else 0:.1f}%")
        print(f"Total time: {time.time() - self.start_time.timestamp():.1f} seconds")
        print(f"Results saved to: {self.config['output_base_dir']}")
        
        return {
            'success': True,
            'total_subjects': self.total_subjects,
            'successful_subjects': self.successful_subjects,
            'failed_subjects': self.failed_subjects,
            'results': self.results
        }


def batch_process_subjects(config=None):
    """
    Convenience function for batch processing subjects.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Processing results
    """
    processor = BatchProcessor(config)
    return processor.run()


def main():
    """
    Main function for command line execution.
    """
    
    # Example configuration - modify as needed
    config = {
        'input_base_dir': '/Users/sachitanand/Lab_PatrickMayo/Projects/Surgical_OpenCV',
        'output_base_dir': './batch_processing_results',
        'subject_folder_pattern': '20*',  # Match folders starting with "20" (like timestamps)
        'skip_existing': True,
        'processing_options': {
            'use_preselected_parameters': False,
            'use_frame_preprocessing': False,
            'use_outer_points': False,
            'show_video': False  # Never show video in batch mode
        }
    }
    
    try:
        results = batch_process_subjects(config)
        
        if results['success']:
            print(f"\nBatch processing completed successfully!")
            return True
        else:
            print(f"\nBatch processing failed: {results.get('error', 'Unknown error')}")
            return False
    
    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error during batch processing: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
