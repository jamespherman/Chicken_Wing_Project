"""
Surgical Skill Analysis via Oculometrics
Chicken Wing Dissection Project - Big Picture Data Analysis

This script implements the complete analysis pipeline for quantifying surgical skill
through eye movement dynamics using pure oculometric features.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class IVTEventClassifier:
    """
    Velocity-Threshold Identification (I-VT) Algorithm
    Classifies gaze samples into fixations, saccades, and other events.

    Reference: Hosp et al., 2021

    Updated to support precomputed angular velocity from 3D gaze direction vectors
    (Physics Update) for more accurate physiological measurements.
    """

    def __init__(self, fixation_threshold=30, saccade_threshold=300):
        """
        Args:
            fixation_threshold: Velocity threshold for fixations (degrees/second)
            saccade_threshold: Velocity threshold for saccades (degrees/second)
        """
        self.fixation_threshold = fixation_threshold
        self.saccade_threshold = saccade_threshold

    def calculate_velocity(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Calculate angular velocity between consecutive gaze samples.

        Note: This method calculates velocity from x/y coordinates. For more
        accurate physics-based angular velocity, use precomputed values from
        3D gaze direction vectors via classify_events_with_precomputed_velocity().

        Args:
            x, y: Gaze coordinates (should be in degrees of visual angle)
            t: Timestamps (in seconds)

        Returns:
            velocities: Angular velocity in degrees/second
        """
        # Calculate differences
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)

        # Avoid division by zero and negative time steps
        dt = np.maximum(dt, 1e-6)

        # Calculate Euclidean distance and velocity
        distance = np.sqrt(dx**2 + dy**2)
        velocity = distance / dt

        # Pad with zero for first sample
        velocity = np.concatenate([[0], velocity])

        return velocity

    def classify_events(self, x: np.ndarray, y: np.ndarray, t: np.ndarray,
                       precomputed_velocity: np.ndarray = None) -> pd.DataFrame:
        """
        Classify each gaze sample into FIXATION, SACCADE, or OTHER.

        Args:
            x, y: Gaze coordinates
            t: Timestamps
            precomputed_velocity: Optional precomputed angular velocity (deg/s)
                                  from 3D gaze direction vectors. If provided,
                                  uses this instead of calculating from x/y.

        Returns:
            DataFrame with columns: x, y, t, velocity, gaze_state
        """
        if precomputed_velocity is not None:
            # Use precomputed angular velocity (Physics Update)
            velocity = np.array(precomputed_velocity)
            # Handle NaN values - treat as OTHER
            velocity = np.nan_to_num(velocity, nan=self.fixation_threshold + 1)
        else:
            # Calculate velocity from x/y coordinates (legacy method)
            velocity = self.calculate_velocity(x, y, t)

        # Classify based on velocity thresholds
        gaze_state = np.empty(len(velocity), dtype=object)
        gaze_state[velocity < self.fixation_threshold] = 'FIXATION'
        gaze_state[velocity >= self.saccade_threshold] = 'SACCADE'
        gaze_state[(velocity >= self.fixation_threshold) &
                   (velocity < self.saccade_threshold)] = 'OTHER'

        df = pd.DataFrame({
            'x': x,
            'y': y,
            't': t,
            'velocity': velocity,
            'gaze_state': gaze_state
        })

        return df

    def classify_from_dataframe(self, df: pd.DataFrame,
                                x_col: str = 'transformed_gaze_x',
                                y_col: str = 'transformed_gaze_y',
                                t_col: str = 'gaze_timestamp',
                                velocity_col: str = 'angular_velocity_deg_s') -> pd.DataFrame:
        """
        Classify events directly from an enhanced gaze DataFrame.

        This method is designed to work with the enhanced final_gaze_data.csv
        that includes precomputed angular velocity.

        Args:
            df: DataFrame with gaze data
            x_col: Name of x coordinate column
            y_col: Name of y coordinate column
            t_col: Name of timestamp column
            velocity_col: Name of precomputed velocity column (if present)

        Returns:
            DataFrame with classified gaze events
        """
        # Check if precomputed velocity is available
        precomputed = None
        if velocity_col in df.columns:
            precomputed = df[velocity_col].values

        return self.classify_events(
            df[x_col].values,
            df[y_col].values,
            df[t_col].values,
            precomputed_velocity=precomputed
        )


class OculometricFeatureExtractor:
    """
    Extracts oculometric features for surgical skill assessment.
    Implements Goals A, B, and C from the analysis plan.
    """
    
    def __init__(self, classified_data: pd.DataFrame):
        """
        Args:
            classified_data: DataFrame with gaze events classified by I-VT
        """
        self.data = classified_data
        self.fixations = classified_data[classified_data['gaze_state'] == 'FIXATION']
        self.saccades = classified_data[classified_data['gaze_state'] == 'SACCADE']
    
    # ==================== GOAL A: Oculometric Efficiency Index ====================
    
    def compute_total_saccade_amplitude(self) -> float:
        """
        Goal A, Metric 1: Total Saccade Amplitude (Scanpath Length)
        
        Hypothesis: Strong positive correlation with Completion Time.
        Novices have 4x the saccadic amplitude of experts (Hosp et al., 2021).
        
        Returns:
            Total scanpath length in degrees
        """
        if len(self.saccades) == 0:
            return 0.0
        
        saccade_data = self.saccades.copy()
        
        # Calculate amplitude of each saccade
        dx = saccade_data['x'].diff()
        dy = saccade_data['y'].diff()
        amplitudes = np.sqrt(dx**2 + dy**2)
        
        total_amplitude = amplitudes.sum()
        return total_amplitude
    
    def compute_fixation_frequency(self) -> float:
        """
        Goal A, Metric 2: Fixation Frequency (Rate)
        
        Hypothesis: High Fixations per Minute rate correlates positively
        with Completion Time (i.e., lower skill).
        
        Reference: Dalveren & Cagiltay (2020) found number of fixations
        differentiates skill groups (p<0.05).
        
        Returns:
            Fixations per minute
        """
        if len(self.data) == 0 or len(self.fixations) == 0:
            return 0.0
        
        # Calculate total duration in minutes
        duration_seconds = self.data['t'].max() - self.data['t'].min()
        duration_minutes = duration_seconds / 60.0
        
        if duration_minutes == 0:
            return 0.0
        
        # Count unique fixation events (consecutive fixations are one event)
        fixation_events = (self.data['gaze_state'] == 'FIXATION') & \
                         (self.data['gaze_state'].shift(1) != 'FIXATION')
        num_fixations = fixation_events.sum()
        
        fixations_per_minute = num_fixations / duration_minutes
        return fixations_per_minute
    
    # ============ GOAL B: Motor Control & Planning Signature ============
    
    def compute_saccade_peak_velocity_std(self) -> float:
        """
        Goal B, Metric 1: Saccade Peak Velocity (Standard Deviation)
        
        Hypothesis: Lower standard deviation correlates with shorter
        Completion Times (higher skill). Experts have more consistent,
        planned saccade speeds (Hosp et al., 2021).
        
        Returns:
            Standard deviation of saccade peak velocities
        """
        if len(self.saccades) == 0:
            return 0.0
        
        saccade_data = self.saccades.copy()
        
        # Group consecutive saccade samples into individual saccade events
        saccade_groups = (saccade_data['gaze_state'] !=
                         saccade_data['gaze_state'].shift(1)).cumsum()
        
        # Find peak velocity for each saccade event
        peak_velocities = saccade_data.groupby(saccade_groups)['velocity'].max()
        
        if len(peak_velocities) < 2:
            return 0.0
        
        std_dev = peak_velocities.std()
        return std_dev
    
    # ================ GOAL C: Cognitive Stability Signature ================
    
    def compute_fixation_duration_distribution(self) -> Dict[str, float]:
        """
        Goal C, Metric 1: Fixation Duration Distribution
        
        Hypothesis: Experts exhibit bimodal distribution (short "scanning"
        fixations and long "working" fixations). Novices show unimodal,
        medium-length distribution indicating continuous searching.
        
        Returns:
            Dictionary with distribution metrics:
            - mean_duration: Mean fixation duration
            - std_duration: Standard deviation
            - bimodality_coefficient: Measure of bimodality (>0.555 suggests bimodal)
            - short_fixation_proportion: Proportion of fixations < 200ms
            - long_fixation_proportion: Proportion of fixations > 500ms
        """
        if len(self.fixations) == 0:
            return {
                'mean_duration': 0.0,
                'std_duration': 0.0,
                'bimodality_coefficient': 0.0,
                'short_fixation_proportion': 0.0,
                'long_fixation_proportion': 0.0
            }
        
        fixation_data = self.fixations.copy()
        
        # Group consecutive fixation samples into individual fixation events
        fixation_groups = (fixation_data['gaze_state'] !=
                          fixation_data['gaze_state'].shift(1)).cumsum()
        
        # Calculate duration of each fixation event
        durations = []
        for group_id in fixation_groups.unique():
            group = fixation_data[fixation_groups == group_id]
            duration = group['t'].max() - group['t'].min()
            durations.append(duration * 1000)  # Convert to milliseconds
        
        durations = np.array(durations)
        
        if len(durations) < 3:
            return {
                'mean_duration': durations.mean() if len(durations) > 0 else 0.0,
                'std_duration': 0.0,
                'bimodality_coefficient': 0.0,
                'short_fixation_proportion': 0.0,
                'long_fixation_proportion': 0.0
            }
        
        # Calculate bimodality coefficient
        # BC = (skewness^2 + 1) / (kurtosis + 3*(n-1)^2/((n-2)*(n-3)))
        # BC > 0.555 suggests bimodal distribution
        n = len(durations)
        skewness = stats.skew(durations)
        kurtosis_excess = stats.kurtosis(durations)  # Excess kurtosis
        
        if n > 3:
            bc_denominator = kurtosis_excess + 3 * (n - 1)**2 / ((n - 2) * (n - 3))
            bimodality_coefficient = (skewness**2 + 1) / bc_denominator
        else:
            bimodality_coefficient = 0.0
        
        # Calculate proportions
        short_threshold = 200  # ms
        long_threshold = 500   # ms
        
        short_proportion = (durations < short_threshold).sum() / len(durations)
        long_proportion = (durations > long_threshold).sum() / len(durations)
        
        return {
            'mean_duration': durations.mean(),
            'std_duration': durations.std(),
            'bimodality_coefficient': bimodality_coefficient,
            'short_fixation_proportion': short_proportion,
            'long_fixation_proportion': long_proportion
        }
    
    def extract_all_features(self) -> Dict[str, float]:
        """
        Extract all oculometric features for surgical skill assessment.
        
        Returns:
            Dictionary containing all features from Goals A, B, and C
        """
        features = {}
        
        # Goal A: Oculometric Efficiency Index
        features['total_saccade_amplitude'] = self.compute_total_saccade_amplitude()
        features['fixation_frequency'] = self.compute_fixation_frequency()
        
        # Goal B: Motor Control & Planning Signature
        features['saccade_peak_velocity_std'] = self.compute_saccade_peak_velocity_std()
        
        # Goal C: Cognitive Stability Signature
        fixation_dist = self.compute_fixation_duration_distribution()
        features.update(fixation_dist)
        
        return features


class SurgicalSkillAnalyzer:
    """
    Complete analysis pipeline for surgical skill assessment via oculometrics.
    Implements the 4-step workflow from the analysis plan.

    Updated to support whole-session analysis mode when task timestamps
    are not available, using recording duration as performance proxy.
    """

    def __init__(self, task_timestamps_path: str = None):
        """
        Args:
            task_timestamps_path: Path to task_timestamps.csv lookup file.
                                  Optional for whole-session analysis mode.
        """
        self.task_timestamps = None
        if task_timestamps_path:
            self.task_timestamps = pd.read_csv(task_timestamps_path)
        self.ivt_classifier = IVTEventClassifier()
        self.results = []
    
    def segment_temporal_data(self, gaze_data: pd.DataFrame,
                              subject_id: str, task_id: str) -> pd.DataFrame:
        """
        Step 1: Temporal Segmentation
        
        Segment raw data using task_timestamps.csv lookup file for
        task-normalized comparison across subjects.
        
        Args:
            gaze_data: Raw gaze data with columns [x, y, t]
            subject_id: Subject identifier
            task_id: Task identifier (e.g., "Task 2: Intramuscular dissection")
            
        Returns:
            Segmented gaze data for the specified task
        """
        # Find task timestamps for this subject and task
        mask = (self.task_timestamps['subject_id'] == subject_id) & \
               (self.task_timestamps['task_id'] == task_id)
        
        if not mask.any():
            raise ValueError(f"No timestamps found for {subject_id}, {task_id}")
        
        task_info = self.task_timestamps[mask].iloc[0]
        start_time = task_info['start_time']
        end_time = task_info['end_time']
        
        # Segment data
        segmented = gaze_data[(gaze_data['t'] >= start_time) &
                             (gaze_data['t'] <= end_time)].copy()
        
        # Normalize time to start at 0
        segmented['t'] = segmented['t'] - start_time
        
        return segmented
    
    def classify_events(self, gaze_data: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Event Classification
        
        Apply I-VT algorithm to classify gaze samples into
        fixations, saccades, and other events.
        
        Args:
            gaze_data: Segmented gaze data
            
        Returns:
            Classified gaze data with event labels
        """
        classified = self.ivt_classifier.classify_events(
            gaze_data['x'].values,
            gaze_data['y'].values,
            gaze_data['t'].values
        )
        
        return classified
    
    def extract_features(self, classified_data: pd.DataFrame) -> Dict[str, float]:
        """
        Step 3: Feature Extraction
        
        Compute metrics from Goals A, B, and C for the subject/task.
        
        Args:
            classified_data: Gaze data with event classifications
            
        Returns:
            Dictionary of oculometric features
        """
        extractor = OculometricFeatureExtractor(classified_data)
        features = extractor.extract_all_features()
        
        return features
    
    def analyze_subject_task(self, gaze_data: pd.DataFrame,
                            subject_id: str, task_id: str,
                            completion_time: float) -> Dict[str, any]:
        """
        Complete analysis for a single subject and task.
        
        Args:
            gaze_data: Raw gaze data
            subject_id: Subject identifier
            task_id: Task identifier
            completion_time: Time to completion (performance metric)
            
        Returns:
            Dictionary with subject_id, task_id, completion_time, and all features
        """
        # Step 1: Temporal Segmentation
        segmented = self.segment_temporal_data(gaze_data, subject_id, task_id)
        
        # Step 2: Event Classification
        classified = self.classify_events(segmented)
        
        # Step 3: Feature Extraction
        features = self.extract_features(classified)
        
        # Combine results
        result = {
            'subject_id': subject_id,
            'task_id': task_id,
            'completion_time': completion_time,
            **features
        }
        
        self.results.append(result)
        return result
    
    def create_feature_matrix(self) -> pd.DataFrame:
        """
        Create feature matrix with rows=subjects and columns=metrics.
        
        Returns:
            DataFrame with all extracted features
        """
        return pd.DataFrame(self.results)
    
    def statistical_analysis(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Statistical Analysis
        
        Correlate Feature Matrix with "Time to Completion" to validate
        which features are strongest predictors of surgical skill.
        
        Args:
            feature_matrix: DataFrame with all features and completion times
            
        Returns:
            DataFrame with correlation results (Pearson and Spearman)
        """
        # Get feature columns (exclude identifiers and completion_time)
        feature_cols = [col for col in feature_matrix.columns
                       if col not in ['subject_id', 'task_id', 'completion_time']]
        
        results = []
        
        for feature in feature_cols:
            # Remove NaN values for correlation
            valid_data = feature_matrix[[feature, 'completion_time']].dropna()
            
            if len(valid_data) < 3:
                continue
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(
                valid_data[feature],
                valid_data['completion_time']
            )
            
            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = stats.spearmanr(
                valid_data[feature],
                valid_data['completion_time']
            )
            
            results.append({
                'feature': feature,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(valid_data)
            })
        
        correlation_results = pd.DataFrame(results)
        
        # Sort by absolute Spearman correlation (most predictive features first)
        correlation_results['abs_spearman_r'] = correlation_results['spearman_r'].abs()
        correlation_results = correlation_results.sort_values(
            'abs_spearman_r', ascending=False
        ).drop('abs_spearman_r', axis=1)
        
        return correlation_results
    
    def generate_report(self, correlation_results: pd.DataFrame) -> str:
        """
        Generate a summary report of the analysis.
        
        Args:
            correlation_results: DataFrame with statistical analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("SURGICAL SKILL ANALYSIS - OCULOMETRIC FEATURES")
        report.append("=" * 80)
        report.append("")
        
        report.append("HYPOTHESIS VALIDATION:")
        report.append("-" * 80)
        
        # Goal A, Hypothesis 1: Total Saccade Amplitude
        tsa = correlation_results[correlation_results['feature'] == 'total_saccade_amplitude']
        if not tsa.empty:
            tsa_r = tsa.iloc[0]['spearman_r']
            tsa_p = tsa.iloc[0]['spearman_p']
            tsa_sig = "SIGNIFICANT" if tsa_p < 0.05 else "Not significant"
            report.append(f"\nGoal A, Hypothesis 1: Total Saccade Amplitude")
            report.append(f"  Expected: Positive correlation with Completion Time")
            report.append(f"  Result: r={tsa_r:.3f}, p={tsa_p:.4f} {tsa_sig}")
            if tsa_r > 0 and tsa_p < 0.05:
                report.append("  Hypothesis SUPPORTED - Higher amplitude → Lower skill")
            else:
                report.append("  Hypothesis NOT supported")
        
        # Goal A, Hypothesis 2: Fixation Frequency
        ff = correlation_results[correlation_results['feature'] == 'fixation_frequency']
        if not ff.empty:
            ff_r = ff.iloc[0]['spearman_r']
            ff_p = ff.iloc[0]['spearman_p']
            ff_sig = "SIGNIFICANT" if ff_p < 0.05 else " Not significant"
            report.append(f"\nGoal A, Hypothesis 2: Fixation Frequency")
            report.append(f"  Expected: Positive correlation with Completion Time")
            report.append(f"  Result: r={ff_r:.3f}, p={ff_p:.4f} {ff_sig}")
            if ff_r > 0 and ff_p < 0.05:
                report.append("  Hypothesis SUPPORTED - Higher frequency → Lower skill")
            else:
                report.append("  Hypothesis NOT supported")
        
        # Goal B, Hypothesis: Saccade Peak Velocity Std
        spv = correlation_results[correlation_results['feature'] == 'saccade_peak_velocity_std']
        if not spv.empty:
            spv_r = spv.iloc[0]['spearman_r']
            spv_p = spv.iloc[0]['spearman_p']
            spv_sig = "SIGNIFICANT" if spv_p < 0.05 else "Not significant"
            report.append(f"\nGoal B, Hypothesis: Saccade Peak Velocity Consistency")
            report.append(f"  Expected: Negative correlation (lower std → shorter time)")
            report.append(f"  Result: r={spv_r:.3f}, p={spv_p:.4f} {spv_sig}")
            if spv_r < 0 and spv_p < 0.05:
                report.append("  Hypothesis SUPPORTED - More consistent → Higher skill")
            else:
                report.append("   Hypothesis NOT supported")
        
        # Goal C, Hypothesis: Fixation Duration Distribution
        bc = correlation_results[correlation_results['feature'] == 'bimodality_coefficient']
        if not bc.empty:
            bc_r = bc.iloc[0]['spearman_r']
            bc_p = bc.iloc[0]['spearman_p']
            bc_sig = " SIGNIFICANT" if bc_p < 0.05 else " Not significant"
            report.append(f"\nGoal C, Hypothesis: Fixation Duration Bimodality")
            report.append(f"  Expected: Negative correlation (higher bimodality → shorter time)")
            report.append(f"  Result: r={bc_r:.3f}, p={bc_p:.4f} {bc_sig}")
            if bc_r < 0 and bc_p < 0.05:
                report.append("  Hypothesis SUPPORTED - Bimodal distribution → Higher skill")
            else:
                report.append("  Hypothesis NOT supported")
        
        report.append("\n" + "=" * 80)
        report.append("TOP PREDICTIVE FEATURES (by |Spearman r|):")
        report.append("-" * 80)
        
        for idx, row in correlation_results.head(5).iterrows():
            sig = "***" if row['spearman_p'] < 0.001 else \
                  "**" if row['spearman_p'] < 0.01 else \
                  "*" if row['spearman_p'] < 0.05 else "ns"
            report.append(f"{row['feature']:35s} r={row['spearman_r']:7.3f}  "
                         f"p={row['spearman_p']:.4f} {sig}")
        
        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def analyze_whole_session(self, gaze_df: pd.DataFrame,
                              subject_id: str) -> Dict[str, any]:
        """
        Analyze an entire recording session without task segmentation.

        This mode is used when task-specific timestamps are not available.
        Uses recording duration as a performance proxy.

        Args:
            gaze_df: DataFrame with enhanced gaze data including:
                     - gaze_timestamp
                     - transformed_gaze_x, transformed_gaze_y
                     - angular_velocity_deg_s (precomputed)
            subject_id: Subject identifier

        Returns:
            Dictionary with subject_id, recording_duration, and all features
        """
        # Filter valid data
        valid_mask = (
            gaze_df['transformed_gaze_x'].notna() &
            gaze_df['transformed_gaze_y'].notna()
        )
        valid_data = gaze_df[valid_mask].copy()

        if len(valid_data) < 100:
            return {
                'subject_id': subject_id,
                'recording_duration': 0,
                'error': 'Insufficient valid data'
            }

        # Calculate recording duration
        recording_duration = (
            valid_data['gaze_timestamp'].max() -
            valid_data['gaze_timestamp'].min()
        )

        # Use precomputed angular velocity if available
        precomputed_velocity = None
        if 'angular_velocity_deg_s' in valid_data.columns:
            precomputed_velocity = valid_data['angular_velocity_deg_s'].values

        # Classify events using I-VT
        classified = self.ivt_classifier.classify_events(
            valid_data['transformed_gaze_x'].values,
            valid_data['transformed_gaze_y'].values,
            valid_data['gaze_timestamp'].values,
            precomputed_velocity=precomputed_velocity
        )

        # Extract features
        extractor = OculometricFeatureExtractor(classified)
        features = extractor.extract_all_features()

        # Compile results
        result = {
            'subject_id': subject_id,
            'recording_duration': recording_duration,
            'total_samples': len(valid_data),
            **features
        }

        self.results.append(result)
        return result
