from pathlib import Path
from ...logging_config import get_logger

logger = get_logger(__name__)

def discover_subject_folders(input_base_dir, video_filename, gaze_filename, subject_folder_pattern, subjects_to_skip):
    """
    Discover subject folders in the input base directory, excluding those in the skip list.

    Returns:
        list: List of valid subject folder paths (excluding skipped subjects)
    """
    input_path = Path(input_base_dir)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return [], 0

    logger.info(f"Scanning for subject folders in: {input_path}")

    # Get the skip list and convert to set for faster lookup
    subjects_to_skip_set = set(subjects_to_skip)
    if subjects_to_skip_set:
        logger.info(f"Skip list contains {len(subjects_to_skip_set)} subjects: {sorted(subjects_to_skip_set)}")

    subject_folders = []
    skipped_count = 0

    for folder_path in input_path.glob(subject_folder_pattern):
        if folder_path.is_dir():
            folder_name = folder_path.name

            if folder_name in subjects_to_skip_set:
                logger.info(f"Skipping {folder_name}: Subject is in skip list")
                skipped_count += 1
                continue

            video_file = folder_path / video_filename
            gaze_file = folder_path / gaze_filename

            if video_file.exists() and gaze_file.exists():
                subject_folders.append(folder_path)
                logger.debug(f"Found valid subject folder: {folder_path.name}")
            else:
                missing_files = []
                if not video_file.exists():
                    missing_files.append(video_filename)
                if not gaze_file.exists():
                    missing_files.append(gaze_filename)
                logger.warning(f"Skipping {folder_path.name}: Missing {', '.join(missing_files)}")

    logger.info(f"Found {len(subject_folders)} valid subject folders")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} subjects due to skip list")

    return sorted(subject_folders), skipped_count
