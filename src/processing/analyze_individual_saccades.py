"""
Detailed analysis of individual saccade events
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surgical_skill_analysis import IVTEventClassifier

# Load your gaze data
gaze_file = '/Users/sachitanand/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofPittsburgh/Herman, James P. - SurgicalCognition/Chicken_Wing_Project/data/processed/20231027T170020Z/20231027T170020Z_final_gaze_data.csv'
print(f"Loading {gaze_file}...")
gaze_data = pd.read_csv(gaze_file)

# Remove NaN values
gaze_data = gaze_data.dropna(subset=['transformed_gaze_x', 'transformed_gaze_y'])
print(f"Valid samples: {len(gaze_data)}")

# Extract coordinates and time
x = gaze_data['transformed_gaze_x'].values
y = gaze_data['transformed_gaze_y'].values
t = gaze_data['gaze_timestamp'].values

# Run I-VT classification
print("\nRunning I-VT classification...")
classifier = IVTEventClassifier(fixation_threshold=30, saccade_threshold=300)
classified = classifier.classify_events(x, y, t)

# Extract saccade events
saccade_data = classified[classified['gaze_state'] == 'SACCADE'].copy()

# Group consecutive saccades into individual events
saccade_indices = saccade_data.index.to_series()
breaks = (saccade_indices.diff() > 1) | (saccade_data.index == saccade_data.index[0])
saccade_data['event_id'] = breaks.cumsum()

# Get first 5 saccade events
event_ids = saccade_data['event_id'].unique()[:5]

print(f"\nAnalyzing first 5 saccade events...")

# Create figure with subplots for each saccade
fig, axes = plt.subplots(5, 4, figsize=(20, 15))
fig.suptitle('Detailed Analysis of 5 Saccade Events', fontsize=16, fontweight='bold')

for idx, event_id in enumerate(event_ids):
    # Extract this saccade event
    event = saccade_data[saccade_data['event_id'] == event_id]
    
    # Get data arrays
    event_x = event['x'].values
    event_y = event['y'].values
    event_t = event['t'].values
    event_vel = event['velocity'].values
    
    # Calculate acceleration (change in velocity)
    # acceleration[i] = (velocity[i+1] - velocity[i-1]) / 2
    event_accel = np.zeros(len(event_vel))
    if len(event_vel) > 2:
        for i in range(1, len(event_vel) - 1):
            event_accel[i] = (event_vel[i+1] - event_vel[i-1]) / 2
        event_accel[0] = event_vel[1] - event_vel[0]  # First point
        event_accel[-1] = event_vel[-1] - event_vel[-2]  # Last point
    
    # Calculate statistics
    start_x, start_y = event_x[0], event_y[0]
    end_x, end_y = event_x[-1], event_y[-1]
    amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    duration = (event_t[-1] - event_t[0]) * 1000  # Convert to ms
    peak_velocity = event_vel.max()
    
    print(f"\nSaccade {idx + 1} (Event ID: {event_id}):")
    print(f"  Samples: {len(event)}")
    print(f"  Duration: {duration:.1f} ms")
    print(f"  Start position: ({start_x:.2f}, {start_y:.2f})")
    print(f"  End position: ({end_x:.2f}, {end_y:.2f})")
    print(f"  Amplitude: {amplitude:.2f} degrees")
    print(f"  Peak velocity: {peak_velocity:.1f} degrees/s")
    print(f"  Mean velocity: {event_vel.mean():.1f} degrees/s")
    print(f"  Peak acceleration: {np.abs(event_accel).max():.1f} degrees/s²")
    
    # Normalize time to start at 0
    event_t_normalized = (event_t - event_t[0]) * 1000  # Convert to ms
    
    # Plot 1: X position over time
    ax = axes[idx, 0]
    ax.plot(event_t_normalized, event_x, 'b-o', linewidth=2, markersize=4)
    ax.axhline(start_x, color='green', linestyle='--', alpha=0.5, label='Start')
    ax.axhline(end_x, color='red', linestyle='--', alpha=0.5, label='End')
    ax.set_xlabel('Time (ms)', fontsize=9)
    ax.set_ylabel('X Position', fontsize=9)
    ax.set_title(f'Saccade {idx+1}: X Position', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Plot 2: Y position over time
    ax = axes[idx, 1]
    ax.plot(event_t_normalized, event_y, 'b-o', linewidth=2, markersize=4)
    ax.axhline(start_y, color='green', linestyle='--', alpha=0.5, label='Start')
    ax.axhline(end_y, color='red', linestyle='--', alpha=0.5, label='End')
    ax.set_xlabel('Time (ms)', fontsize=9)
    ax.set_ylabel('Y Position', fontsize=9)
    ax.set_title(f'Saccade {idx+1}: Y Position', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Plot 3: Velocity over time
    ax = axes[idx, 2]
    ax.plot(event_t_normalized, event_vel, 'r-o', linewidth=2, markersize=4)
    ax.axhline(peak_velocity, color='orange', linestyle='--', alpha=0.5, label=f'Peak: {peak_velocity:.0f}°/s')
    ax.set_xlabel('Time (ms)', fontsize=9)
    ax.set_ylabel('Velocity (deg/s)', fontsize=9)
    ax.set_title(f'Saccade {idx+1}: Velocity Profile', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Plot 4: Acceleration over time
    ax = axes[idx, 3]
    ax.plot(event_t_normalized, event_accel, 'g-o', linewidth=2, markersize=4)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (ms)', fontsize=9)
    ax.set_ylabel('Acceleration (deg/s²)', fontsize=9)
    ax.set_title(f'Saccade {idx+1}: Acceleration', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('detailed_saccade_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: detailed_saccade_analysis.png")
plt.show()

# Create a 2D trajectory plot for all 5 saccades
fig, ax = plt.subplots(figsize=(12, 10))
colors = ['blue', 'red', 'green', 'orange', 'purple']

for idx, event_id in enumerate(event_ids):
    event = saccade_data[saccade_data['event_id'] == event_id]
    event_x = event['x'].values
    event_y = event['y'].values
    
    # Plot trajectory
    ax.plot(event_x, event_y, '-o', color=colors[idx], linewidth=2,
            markersize=6, label=f'Saccade {idx+1}', alpha=0.7)
    
    # Mark start and end
    ax.plot(event_x[0], event_y[0], 'o', color=colors[idx],
            markersize=12, markerfacecolor='white', markeredgewidth=2)
    ax.plot(event_x[-1], event_y[-1], 's', color=colors[idx],
            markersize=12, markerfacecolor=colors[idx], markeredgewidth=2)

ax.set_xlabel('X Position (degrees)', fontsize=12)
ax.set_ylabel('Y Position (degrees)', fontsize=12)
ax.set_title('2D Trajectories of 5 Saccades\n(○ = start, ■ = end)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.savefig('saccade_trajectories_2d.png', dpi=150, bbox_inches='tight')
print("Saved: saccade_trajectories_2d.png")
plt.show()

print("\nAnalysis complete!")
