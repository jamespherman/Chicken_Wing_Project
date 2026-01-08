"""
Test I-VT classification on real gaze data without task segmentation.
Validates by plotting the saccadic main sequence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surgical_skill_analysis import IVTEventClassifier

# Load your gaze data directly
gaze_file = '/Users/sachitanand/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofPittsburgh/Herman, James P. - SurgicalCognition/Chicken_Wing_Project/data/processed/20231027T170020Z/20231027T170020Z_final_gaze_data.csv'
print(f"Loading {gaze_file}...")
gaze_data = pd.read_csv(gaze_file)

print(f"Loaded {len(gaze_data)} samples")
print(f"Columns: {gaze_data.columns.tolist()}")
print(f"First few rows:\n{gaze_data.head()}")

# Remove rows with NaN gaze coordinates
print(f"\nRemoving NaN values...")
gaze_data = gaze_data.dropna(subset=['transformed_gaze_x', 'transformed_gaze_y'])
print(f"Valid samples after removing NaN: {len(gaze_data)}")

# Extract coordinates and time with correct column names
x = gaze_data['transformed_gaze_x'].values
y = gaze_data['transformed_gaze_y'].values
t = gaze_data['gaze_timestamp'].values

# Run I-VT classification
print("\nRunning I-VT classification...")
classifier = IVTEventClassifier(fixation_threshold=30, saccade_threshold=300)
classified = classifier.classify_events(x, y, t)

# Print classification results
fixations = classified[classified['gaze_state'] == 'FIXATION']
saccades = classified[classified['gaze_state'] == 'SACCADE']
other = classified[classified['gaze_state'] == 'OTHER']

print(f"\nClassification Results:")
print(f"  Fixations: {len(fixations)} samples ({len(fixations)/len(classified)*100:.1f}%)")
print(f"  Saccades: {len(saccades)} samples ({len(saccades)/len(classified)*100:.1f}%)")
print(f"  Other: {len(other)} samples ({len(other)/len(classified)*100:.1f}%)")

# Extract saccade events (group consecutive saccade samples)
print("\nExtracting saccade events...")
saccade_data = classified[classified['gaze_state'] == 'SACCADE'].copy()

if len(saccade_data) > 0:
    # Group consecutive saccades into individual events
    # Use the original index to detect non-consecutive samples
    saccade_indices = saccade_data.index.to_series()
    breaks = (saccade_indices.diff() > 1) | (saccade_data.index == saccade_data.index[0])
    saccade_data['event_id'] = breaks.cumsum()
    
    # Calculate amplitude and peak velocity for each saccade
    saccade_events = []
    for event_id in saccade_data['event_id'].unique():
        event = saccade_data[saccade_data['event_id'] == event_id]
        
        # Skip very short events (likely noise)
        if len(event) < 2:
            continue
        
        # Amplitude: distance from start to end of saccade
        start_x, start_y = event.iloc[0]['x'], event.iloc[0]['y']
        end_x, end_y = event.iloc[-1]['x'], event.iloc[-1]['y']
        amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Peak velocity
        peak_velocity = event['velocity'].max()
        
        # Duration
        duration = event['t'].max() - event['t'].min()
        
        saccade_events.append({
            'amplitude': amplitude,
            'peak_velocity': peak_velocity,
            'duration': duration
        })
    
    saccade_events_df = pd.DataFrame(saccade_events)
    print(f"Found {len(saccade_events_df)} saccade events")
    
    if len(saccade_events_df) > 0:
        print(f"\nSaccade statistics:")
        print(f"  Amplitude: {saccade_events_df['amplitude'].mean():.2f}° ± {saccade_events_df['amplitude'].std():.2f}°")
        print(f"  Peak velocity: {saccade_events_df['peak_velocity'].mean():.1f}°/s ± {saccade_events_df['peak_velocity'].std():.1f}°/s")
        print(f"  Duration: {saccade_events_df['duration'].mean()*1000:.1f}ms ± {saccade_events_df['duration'].std()*1000:.1f}ms")
        
        # Plot the Main Sequence
        print("\nPlotting saccadic main sequence...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Amplitude vs Peak Velocity (classic main sequence)
        ax = axes[0]
        ax.scatter(saccade_events_df['amplitude'],
                   saccade_events_df['peak_velocity'],
                   alpha=0.5, s=30)
        ax.set_xlabel('Saccade Amplitude (degrees)', fontsize=12)
        ax.set_ylabel('Peak Velocity (degrees/s)', fontsize=12)
        ax.set_title('Saccadic Main Sequence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Amplitude vs Duration
        ax = axes[1]
        ax.scatter(saccade_events_df['amplitude'],
                   saccade_events_df['duration']*1000,
                   alpha=0.5, s=30, color='coral')
        ax.set_xlabel('Saccade Amplitude (degrees)', fontsize=12)
        ax.set_ylabel('Duration (ms)', fontsize=12)
        ax.set_title('Amplitude-Duration Relationship', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('main_sequence_validation.png', dpi=150, bbox_inches='tight')
        print("Saved: main_sequence_validation.png")
        plt.show()
    else:
        print("WARNING: No valid saccade events after filtering!")
    
else:
    print("WARNING: No saccades detected! Check your data or thresholds.")

# Save classified data for inspection
classified.to_csv('classified_gaze_data.csv', index=False)
print("\nSaved: classified_gaze_data.csv")
