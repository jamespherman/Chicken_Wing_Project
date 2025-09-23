import json
import time
from datetime import datetime
from pathlib import Path

def save_processing_log(result, log_path):
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

def create_summary_report(config, results, skipped_subjects, total_subjects, successful_subjects, failed_subjects, start_time, logs_dir, figures_dir, processed_data_dir):
    """
    Create an enhanced summary report of all processed subjects with timestamps and skip list info.
    """
    if not config['create_summary_report']:
        return

    print(f"\nCreating enhanced summary report...")

    # Create timestamped filenames
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary_json_path = logs_dir / f"batch_summary_{timestamp}.json"
    summary_txt_path = logs_dir / f"batch_summary_{timestamp}.txt"

    # Calculate additional statistics
    heatmap_successes = sum(1 for r in results
                           if r.get('step3_stats', {}).get('success', False))

    total_gaze_points = sum(r.get('step3_stats', {}).get('statistics', {}).get('filtered_samples', 0)
                           for r in results if r.get('step3_stats'))

    # Count different types of skipped subjects
    output_exists_skipped = sum(1 for r in results
                               if r['status'] == 'skipped' and r.get('reason') == 'Outputs already exist')

    summary = {
        'processing_session': {
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': datetime.now().isoformat(),
            'total_duration': time.time() - start_time.timestamp() if start_time else 0,
            'config': config
        },
        'overall_stats': {
            'total_subjects': total_subjects,
            'successful_subjects': successful_subjects,
            'failed_subjects': failed_subjects,
            'skipped_subjects_skip_list': skipped_subjects,
            'skipped_subjects_output_exists': output_exists_skipped,
            'total_skipped_subjects': skipped_subjects + output_exists_skipped,
            'success_rate': (successful_subjects / total_subjects * 100) if total_subjects > 0 else 0,

            # Skip list information
            'skip_list_enabled': len(config.get('subjects_to_skip', [])) > 0,
            'skip_list_count': len(config.get('subjects_to_skip', [])),
            'skip_list_subjects': config.get('subjects_to_skip', []),

            # Heatmap-specific statistics
            'heatmap_generation_enabled': config['generate_heatmaps'],
            'heatmap_successes': heatmap_successes,
            'heatmap_success_rate': (heatmap_successes / total_subjects * 100) if total_subjects > 0 else 0,
            'total_gaze_points_analyzed': total_gaze_points
        },
        'subject_results': results
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
            f.write(f"Total subjects discovered: {total_subjects + skipped_subjects}\n")
            f.write(f"Subjects processed: {total_subjects}\n")
            f.write(f"Successful (all steps): {successful_subjects}\n")
            f.write(f"Failed: {failed_subjects}\n")
            f.write(f"Skipped (skip list): {skipped_subjects}\n")
            f.write(f"Skipped (existing outputs): {output_exists_skipped}\n")
            f.write(f"Overall success rate: {summary['overall_stats']['success_rate']:.1f}%\n\n")

            # Skip list information
            if skipped_subjects > 0:
                f.write("SKIP LIST INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Skip list enabled: Yes\n")
                f.write(f"Subjects in skip list: {len(config.get('subjects_to_skip', []))}\n")
                skip_list = config.get('subjects_to_skip', [])
                if skip_list:
                    f.write(f"Skip list contents: {', '.join(sorted(skip_list))}\n")
                f.write(f"Subjects actually skipped: {skipped_subjects}\n\n")

            if config['generate_heatmaps']:
                f.write(f"Heatmap visualizations created: {heatmap_successes}/{total_subjects}\n")
                f.write(f"Heatmap success rate: {summary['overall_stats']['heatmap_success_rate']:.1f}%\n")
                f.write(f"Total gaze points analyzed: {total_gaze_points:,}\n\n")

            f.write("OUTPUT DIRECTORY STRUCTURE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"All images (.png): {figures_dir}\n")
            f.write(f"All logs (.txt, .json): {logs_dir}\n")
            f.write(f"All processed data: {processed_data_dir}\n\n")

            f.write("INDIVIDUAL RESULTS:\n")
            f.write("-" * 40 + "\n")
            for result in results:
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
