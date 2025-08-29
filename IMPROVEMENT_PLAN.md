# Improvement Plan

This document outlines the plan for improving the codebase, focusing on robustness, portability, configurability, and maintainability.

## 1. Address Hardcoded Paths

**Issue:** The main batch processing script `src/batch_process_with_heatmaps.py` contains hardcoded, user-specific absolute paths (e.g., `/Users/sachitanand/...`). This makes the script non-portable and difficult for others to run without modification.

**Improvement:**
- The script has been modified to determine the project's root directory dynamically.
- All input and output paths are now constructed relative to the project root, ensuring the script can be run in any environment without manual path adjustments.

## 2. Refactor and Reduce Code Duplication

**Issue:** The `order_points` function was duplicated in `src/processing/gaze_on_perspective_corrected_frames_refactored.py`, leading to code redundancy and potential maintenance issues.

**Improvement:**
- A new shared utility module, `src/processing/utils.py`, has been created.
- The `order_points` function has been moved to this utility module.
- The `gaze_on_perspective_corrected_frames_refactored.py` script has been updated to import the function from the new module, eliminating the duplicated code.

## 3. Centralize Configuration

**Issue:** Several key parameters, such as filenames (`scenevideo.mp4`, `gazedata.gz`), marker IDs, and image dimensions, were hardcoded within the processing and analysis scripts. This made the pipeline inflexible and difficult to adapt.

**Improvement:**
- All hardcoded parameters have been moved to the main configuration dictionary in `src/batch_process_with_heatmaps.py`.
- The processing and analysis scripts now receive these parameters from the main configuration, making the pipeline highly flexible and easy to configure for different datasets or experimental setups.
- The `main` functions in the processing and analysis scripts, which contained hardcoded paths and parameters, have been removed to ensure all processing is handled through the main batch processing script.

## 4. Enhance Output and Reporting

**Issue:** The output directory structure was flat, and the summary reports could be more detailed and better organized.

**Improvement:**
- The output directory structure has been enhanced to be more organized. Subdirectories are now created for each subject within the `processed`, `figures`, and `logs` directories.
- The summary reports have been improved to be more informative, with standardized naming conventions and more detailed statistics.
- The final summary printout in the main script has been updated to be more consistent and readable.
