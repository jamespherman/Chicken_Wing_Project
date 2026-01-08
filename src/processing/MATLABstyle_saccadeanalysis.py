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
gaze_file = '/Users/sachitanand/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofPittsburgh/Herman, James P. - SurgicalCognition/Chicken_Wing_Project/data/processed/20231027T170020Z/20231027T170020Z_final_gaze_data.csv'
print(f"Loading {gaze_file}...")
gaze_data = pd.read_csv(gaze_file)

gaze_data = gaze_data.dropna(subset=['transformed_gaze_x', 'transformed_gaze_y'])
print(f"Valid samples: {len(gaze_data)}")

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
event_ids = sorted(saccade_data["event_id"].unique()[:num_events_to_plot])


# ---------------------------------------------------------
# 6. PLOT ANALYSIS FOR FIRST 5 EVENTS
# ---------------------------------------------------------
print(f"\nAnalyzing first {num_events_to_plot} saccades…")

fig, axes = plt.subplots(num_events_to_plot, 4, figsize=(20, 3*num_events_to_plot))
if num_events_to_plot == 1:
    axes = axes.reshape(1, -1)
    
fig.suptitle('Detailed Analysis of Saccade Events (IVT + Robust Velocity + Filtering)',
             fontsize=16, fontweight='bold')

for idx, event_id in enumerate(event_ids):
    event = saccade_data[saccade_data['event_id'] == event_id]

    event_x = event['transformed_gaze_x'].values
    event_y = event['transformed_gaze_y'].values
    event_t = event['gaze_timestamp'].values
    event_vel = event['velocity'].values

    # Compute acceleration
    event_accel = np.gradient(event_vel)

    # Metrics
    amplitude = np.sqrt((event_x[-1] - event_x[0])**2 + (event_y[-1] - event_y[0])**2)
    duration = (event_t[-1] - event_t[0]) * 1000
    peak_vel = np.max(event_vel)

    print(f"\nSaccade {idx+1}:")
    print(f"  Samples: {len(event)}")
    print(f"  Duration: {duration:.1f} ms")
    print(f"  Amplitude: {amplitude:.2f}°")
    print(f"  Peak velocity: {peak_vel:.1f}°/s")

    # Normalize time
    tn = (event_t - event_t[0]) * 1000

    # ---- X ----
    ax = axes[idx, 0]
    ax.plot(tn, event_x, 'b-o', markersize=4)
    ax.set_title(f"Event {idx+1}: X Position")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("X (°)")
    ax.grid(alpha=0.3)

    # ---- Y ----
    ax = axes[idx, 1]
    ax.plot(tn, event_y, 'b-o', markersize=4)
    ax.set_title(f"Event {idx+1}: Y Position")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Y (°)")
    ax.grid(alpha=0.3)

    # ---- Velocity ----
    ax = axes[idx, 2]
    ax.plot(tn, event_vel, 'r-o', markersize=4)
    ax.axhline(peak_vel, color='orange', linestyle='--', label=f'Peak: {peak_vel:.0f}°/s')
    ax.set_title(f"Event {idx+1}: Velocity")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity (°/s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- Acceleration ----
    ax = axes[idx, 3]
    ax.plot(tn, event_accel, 'g-o', markersize=4)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f"Event {idx+1}: Acceleration")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Accel (°/s²)")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("detailed_saccade_analysis.png", dpi=150)
plt.show()

print("\nSaved detailed_saccade_analysis.png")


# ---------------------------------------------------------
# 7. 2D TRAJECTORIES
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))
colors = ['blue', 'red', 'green', 'orange', 'purple']

for idx, event_id in enumerate(event_ids):
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
