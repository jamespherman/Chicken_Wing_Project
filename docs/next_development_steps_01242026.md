# Next Development Steps (01/24/2026)

This document outlines the immediate technical steps required to enable the "Big-Picture Data Analysis" for the Chicken Wing Dissection Project. The current codebase has the core analysis logic (`SurgicalSkillAnalyzer`) but lacks the necessary data infrastructure and orchestration to run it on the dataset.

## 1. Data Infrastructure: `task_timestamps.csv`

The analysis pipeline relies on temporal segmentation to compare performance across specific tasks (e.g., "Intramuscular dissection"). This requires a lookup file that maps each subject's recording to specific start and end times for each task.

**Action:** Create `data/task_timestamps.csv` (or `data/metadata/task_timestamps.csv`).

**Schema:**
```csv
subject_id,task_id,start_time,end_time,completion_time,notes
```

- `subject_id`: Matching the folder name in `data/processed/` (e.g., `20231027T170020Z`).
- `task_id`: Identifier for the sub-task (e.g., `Task_1`, `Task_2`, `Task_3`, `Task_4`).
- `start_time`: Timestamp (seconds) relative to the start of the gaze recording.
- `end_time`: Timestamp (seconds) relative to the start of the gaze recording.
- `completion_time`: Performance metric (seconds), typically `end_time - start_time`.
- `notes`: Any relevant observations.

**Implementation Step:**
- Create the file with a header row.
- If manual annotation is required, provide a template.
- If logs exist, write a script to populate this automatically.

## 2. Refactor `SurgicalSkillAnalyzer`

The existing `src/processing/surgical_skill_analysis.py` expects column names (`x`, `y`, `t`) that differ from the actual output of the processing pipeline (`transformed_gaze_x`, `transformed_gaze_y`, `gaze_timestamp`).

**Action:** Update `src/processing/surgical_skill_analysis.py`.

**Changes:**
- Update `IVTEventClassifier` or the `SurgicalSkillAnalyzer.classify_events` method to map the columns correctly:
    - `transformed_gaze_x` -> `x`
    - `transformed_gaze_y` -> `y`
    - `gaze_timestamp` -> `t`
- Add robust error handling to skip subjects or tasks if data is missing or invalid (e.g., NaN values).
- Ensure `completion_time` is correctly passed or calculated from the timestamps if not provided explicitly.

## 3. Orchestration Script: `run_surgical_analysis.py`

There is currently no script to run the analysis across the entire dataset.

**Action:** Create `src/analysis/run_surgical_analysis.py`.

**Logic:**
1.  **Load Metadata:** Read `task_timestamps.csv`.
2.  **Discover Data:** Iterate through all subject folders in `data/processed/`.
3.  **Match & Process:**
    - For each subject found in `data/processed/`:
        - specific tasks defined for this subject in `task_timestamps.csv`.
        - Load `final_gaze_data.csv`.
        - Instantiate `SurgicalSkillAnalyzer`.
        - Call `analyze_subject_task` for each task.
4.  **Aggregate & Report:**
    - Collect all results into a Feature Matrix (DataFrame).
    - Run `statistical_analysis` (correlations).
    - Generate and save the text report (`analysis_report.txt`) and the Feature Matrix (`feature_matrix.csv`) to `reports/`.

## 4. Verification and Testing

**Action:** Verify the pipeline with existing data.

- Use the sample data in `data/processed/20231027T170020Z/`.
- Manually create a dummy entry in `task_timestamps.csv` for this subject (e.g., define a 10-second "task" based on the file's timestamp range).
- Run `python src/analysis/run_surgical_analysis.py`.
- Verify that `reports/analysis_report.txt` is generated and contains correlation results (even if not statistically significant due to low N).
