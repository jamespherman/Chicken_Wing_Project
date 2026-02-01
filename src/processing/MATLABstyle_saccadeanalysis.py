"""
Detailed analysis of individual saccade events
Using MATLAB-style robust velocity and IVT saccade classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ---------------------------------------------------------
# 1. MATLAB-STYLE ROBUST VELOCITY
# ---------------------------------------------------------
def smoothdiff_robust_python(position, fs=100, window=31, polyorder=3):
    """Savitzky–Golay derivative (MATLAB-like robust differentiation)."""
    if window % 2 == 0:
        window += 1
    deriv = savgol_filter(position, window, polyorder, deriv=1)
    return deriv * fs  # units = deg/s


# ---------------------------------------------------------
# 2. IVT SACCADE CLASSIFIER WITH FILTERING
# ---------------------------------------------------------
class IVTEventClassifier:
    def __init__(self, velocity_threshold=100,
                 min_duration_ms=10,
                 max_duration_ms=120,
                 max_amplitude=50,
                 max_velocity=800):
        self.velocity_threshold = velocity_threshold
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.max_amplitude = max_amplitude
        self.max_velocity = max_velocity

    def classify(self, timestamps, velocity, x, y):
        above = velocity > self.velocity_threshold
        events = []
        n = len(velocity)
        i = 0

        while i < n:
            if above[i]:
                start_idx = i
                while i < n and above[i]:
                    i += 1
                end_idx = i - 1

                # Calculate metrics
                duration_ms = (timestamps[end_idx] - timestamps[start_idx]) * 1000
                amplitude = np.sqrt((x[end_idx] - x[start_idx])**2 +
                                   (y[end_idx] - y[start_idx])**2)
                peak_velocity = float(np.max(velocity[start_idx:end_idx + 1]))

                # Apply filters (INCLUDING max velocity)
                if (self.min_duration_ms <= duration_ms <= self.max_duration_ms and
                    amplitude <= self.max_amplitude and
                    peak_velocity <= self.max_velocity):
                    
                    events.append({
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "start_time": timestamps[start_idx],
                        "end_time": timestamps[end_idx],
                        "duration_ms": duration_ms,
                        "amplitude": amplitude,
                        "peak_velocity": peak_velocity
                    })
            else:
                i += 1

        return events


# ---------------------------------------------------------
# 3. LOAD DATA
# ---------------------------------------------------------
gaze_file = '/Users/sachitanand/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofPittsburgh/Herman, James P. - SurgicalCognition/Chicken_Wing_Project/data/processed/20231012T122519Z/20231012T122519Z_final_gaze_data.csv'
print(f"Loading {gaze_file}...")
gaze_data = pd.read_csv(gaze_file)

print(f"\nTotal rows loaded: {len(gaze_data)}")
print(f"\nColumn names in CSV:")
print(gaze_data.columns.tolist())

print(f"\nFirst few rows:")
print(gaze_data.head())

print(f"\nData types:")
print(gaze_data.dtypes)

# Check for the expected columns
expected_cols = ['transformed_gaze_x', 'transformed_gaze_y', 'gaze_timestamp']
missing_cols = [col for col in expected_cols if col not in gaze_data.columns]

if missing_cols:
    print(f"\nERROR: Missing expected columns: {missing_cols}")
    print("\nPlease check your CSV file column names.")
    exit()

# Check for NaN values before dropping
print(f"\nNaN counts before dropping:")
print(f"  transformed_gaze_x: {gaze_data['transformed_gaze_x'].isna().sum()}")
print(f"  transformed_gaze_y: {gaze_data['transformed_gaze_y'].isna().sum()}")
print(f"  gaze_timestamp: {gaze_data['gaze_timestamp'].isna().sum()}")

gaze_data = gaze_data.dropna(subset=['transformed_gaze_x', 'transformed_gaze_y'])
print(f"\nValid samples after dropping NaNs: {len(gaze_data)}")

if len(gaze_data) == 0:
    print("\nERROR: No valid data remaining after dropping NaNs!")
    print("All values in transformed_gaze_x and/or transformed_gaze_y are NaN.")
    exit()

x = gaze_data['transformed_gaze_x'].values
y = gaze_data['transformed_gaze_y'].values
t = gaze_data['gaze_timestamp'].values

fs = int(round(1 / np.median(np.diff(t))))
print(f"Estimated sample rate: {fs} Hz")


# ---------------------------------------------------------
# 4. COMPUTE VELOCITY
# ---------------------------------------------------------
print("Computing robust velocity…")
xVel = smoothdiff_robust_python(x, fs=fs)
yVel = smoothdiff_robust_python(y, fs=fs)

vel = np.sqrt(xVel**2 + yVel**2)
gaze_data["velocity"] = vel


# ---------------------------------------------------------
# 5. APPLY IVT CLASSIFIER WITH FILTERS
# ---------------------------------------------------------
vThresh = 100
print(f"Running IVT classifier @ {vThresh} deg/s…")

ivt = IVTEventClassifier(
    velocity_threshold=vThresh,
    min_duration_ms=10,      # Minimum saccade duration
    max_duration_ms=120,     # Maximum saccade duration
    max_amplitude=50,        # Maximum realistic amplitude in degrees
    max_velocity=800         # Maximum realistic peak velocity (filters artifacts)
)

ivt_events = ivt.classify(timestamps=t, velocity=vel, x=x, y=y)

print(f"Detected {len(ivt_events)} valid saccades (after filtering)")

if len(ivt_events) == 0:
    print("\nNo valid saccades detected!")
    print("Try adjusting thresholds:")
    print("  - Lower velocity_threshold (currently 100)")
    print("  - Increase max_duration_ms (currently 120)")
    print("  - Increase max_amplitude (currently 50)")
    print("  - Increase max_velocity (currently 800)")
    exit()

# Convert IVT events into a dataframe
saccade_rows = []

for event_id, ev in enumerate(ivt_events):
    idxs = range(ev["start_idx"], ev["end_idx"] + 1)
    for idx in idxs:
        saccade_rows.append({
            "event_id": event_id,
            "index": idx
        })

saccade_index_df = pd.DataFrame(saccade_rows)
saccade_data = gaze_data.iloc[saccade_index_df["index"]].copy()
saccade_data["event_id"] = saccade_index_df["event_id"].values

# Select first 5 events
num_events_to_plot = min(5, len(ivt_events))


# ---------------------------------------------------------
# 6. PLOT ANALYSIS FOR FIRST 5 EVENTS (TRANSPOSED + 500ms WINDOW)
# ---------------------------------------------------------
print(f"\nAnalyzing first {num_events_to_plot} saccades…")

# TRANSPOSED: 4 rows (metrics) × num_events columns
fig, axes = plt.subplots(4, num_events_to_plot, figsize=(4*num_events_to_plot, 12))
if num_events_to_plot == 1:
    axes = axes.reshape(-1, 1)
    
fig.suptitle('Detailed Analysis of Saccade Events (IVT + Robust Velocity + Filtering)\n500ms Windows',
             fontsize=16, fontweight='bold')

for col_idx in range(num_events_to_plot):
    event_info = ivt_events[col_idx]
    
    # Extract 500ms window: 250ms before saccade start to 250ms after
    start_time = event_info['start_time']
    window_start = start_time - 0.25  # 250ms before
    window_end = start_time + 0.25    # 250ms after
    
    # Get all data in this window
    mask = (gaze_data['gaze_timestamp'] >= window_start) & (gaze_data['gaze_timestamp'] <= window_end)
    window_data = gaze_data[mask].copy()
    
    if len(window_data) == 0:
        print(f"Warning: No data in window for event {col_idx}")
        continue
    
    # Mark which samples are part of the actual saccade
    saccade_mask = (window_data.index >= event_info['start_idx']) & (window_data.index <= event_info['end_idx'])
    
    # Extract data
    window_x = window_data['transformed_gaze_x'].values
    window_y = window_data['transformed_gaze_y'].values
    window_t = window_data['gaze_timestamp'].values
    window_vel = window_data['velocity'].values
    
    # Compute acceleration for the window
    window_accel = np.gradient(window_vel)
    
    # Normalize time to ms from saccade start
    tn = (window_t - start_time) * 1000
    
    # Get saccade portion for metrics
    saccade_portion = window_data[saccade_mask]
    amplitude = event_info['amplitude']
    duration = event_info['duration_ms']
    peak_vel = event_info['peak_velocity']
    
    print(f"\nSaccade {col_idx+1}:")
    print(f"  Window samples: {len(window_data)}")
    print(f"  Saccade samples: {saccade_mask.sum()}")
    print(f"  Duration: {duration:.1f} ms")
    print(f"  Amplitude: {amplitude:.2f}°")
    print(f"  Peak velocity: {peak_vel:.1f}°/s")

    # ---- Row 0: X Position ----
    ax = axes[0, col_idx]
    ax.plot(tn, window_x, 'b-', linewidth=1, alpha=0.5, label='Window')
    ax.plot(tn[saccade_mask], window_x[saccade_mask], 'b-o', markersize=4, linewidth=2, label='Saccade')
    ax.set_title(f"Saccade {col_idx+1}\n{duration:.1f}ms, {amplitude:.1f}°")
    ax.set_ylabel("X Position (°)")
    ax.grid(alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    if col_idx == 0:
        ax.legend(fontsize=8)

    # ---- Row 1: Y Position ----
    ax = axes[1, col_idx]
    ax.plot(tn, window_y, 'b-', linewidth=1, alpha=0.5)
    ax.plot(tn[saccade_mask], window_y[saccade_mask], 'b-o', markersize=4, linewidth=2)
    ax.set_ylabel("Y Position (°)")
    ax.grid(alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # ---- Row 2: Velocity ----
    ax = axes[2, col_idx]
    ax.plot(tn, window_vel, 'r-', linewidth=1, alpha=0.5)
    ax.plot(tn[saccade_mask], window_vel[saccade_mask], 'r-o', markersize=4, linewidth=2)
    ax.axhline(peak_vel, color='orange', linestyle='--', alpha=0.7, label=f'Peak: {peak_vel:.0f}°/s')
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Threshold')
    ax.set_ylabel("Velocity (°/s)")
    ax.grid(alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    if col_idx == 0:
        ax.legend(fontsize=8)

    # ---- Row 3: Acceleration ----
    ax = axes[3, col_idx]
    ax.plot(tn, window_accel, 'g-', linewidth=1, alpha=0.5)
    ax.plot(tn[saccade_mask], window_accel[saccade_mask], 'g-o', markersize=4, linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Acceleration (°/s²)")
    ax.set_xlabel("Time from saccade start (ms)")
    ax.grid(alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    # Set x-axis limits to exactly -250 to +250 ms
    ax.set_xlim(-250, 250)
    axes[0, col_idx].set_xlim(-250, 250)
    axes[1, col_idx].set_xlim(-250, 250)
    axes[2, col_idx].set_xlim(-250, 250)

plt.tight_layout()
plt.savefig("detailed_saccade_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved detailed_saccade_analysis.png")


# ---------------------------------------------------------
# 7. 2D TRAJECTORIES
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))
colors = ['blue', 'red', 'green', 'orange', 'purple']

for idx in range(num_events_to_plot):
    event_id = idx
    event = saccade_data[saccade_data['event_id'] == event_id]
    ev_info = ivt_events[event_id]
    ax.plot(event['transformed_gaze_x'], event['transformed_gaze_y'],
            '-o', color=colors[idx % len(colors)],
            label=f"S{idx+1}: {ev_info['duration_ms']:.0f}ms, {ev_info['amplitude']:.1f}°",
            markersize=6)
    
    # Mark start and end
    ax.plot(event['transformed_gaze_x'].iloc[0], event['transformed_gaze_y'].iloc[0],
            'o', color=colors[idx % len(colors)], markersize=10, markerfacecolor='white')
    ax.plot(event['transformed_gaze_x'].iloc[-1], event['transformed_gaze_y'].iloc[-1],
            's', color=colors[idx % len(colors)], markersize=10)

ax.set_title("2D Trajectories of Saccades (○=start, □=end)")
ax.set_xlabel("X Position (°)")
ax.set_ylabel("Y Position (°)")
ax.legend()
ax.grid(alpha=0.3)
ax.axis("equal")

plt.tight_layout()
plt.savefig("saccade_trajectories_2d.png", dpi=150)
plt.show()

print("\nSaved saccade_trajectories_2d.png")


# ---------------------------------------------------------
# 8. SUMMARY STATISTICS
# ---------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

durations = [ev['duration_ms'] for ev in ivt_events]
amplitudes = [ev['amplitude'] for ev in ivt_events]
peak_vels = [ev['peak_velocity'] for ev in ivt_events]

print(f"Total saccades detected: {len(ivt_events)}")
print(f"\nDuration (ms):")
print(f"  Mean: {np.mean(durations):.1f} ± {np.std(durations):.1f}")
print(f"  Range: {np.min(durations):.1f} - {np.max(durations):.1f}")
print(f"\nAmplitude (°):")
print(f"  Mean: {np.mean(amplitudes):.2f} ± {np.std(amplitudes):.2f}")
print(f"  Range: {np.min(amplitudes):.2f} - {np.max(amplitudes):.2f}")
print(f"\nPeak Velocity (°/s):")
print(f"  Mean: {np.mean(peak_vels):.1f} ± {np.std(peak_vels):.1f}")
print(f"  Range: {np.min(peak_vels):.1f} - {np.max(peak_vels):.1f}")

print("\nAnalysis complete!")
