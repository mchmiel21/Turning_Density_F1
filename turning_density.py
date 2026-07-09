"""
Computes and plots "Turning Density" charts for Formula 1 circuits of 
any given year based on telemetry data from all qualifying push laps.

"Turning Density" is the cumulative absolute change in vehicle path heading 
divided by the total lap distance traveled.

Published July 2026

@author: mchmiel21
"""
# Select which year's data to run
year = 2026

# import necessary modules
import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import seaborn as sns
import time
import logging
from pathlib import Path

######## COMPUTE TURNING DENSITY METRICS! ########
# Enable caching (so you don't re-download data each time)
CACHE_DIR = Path("f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))
# Suppress terminal logging
fastf1.set_log_level(logging.ERROR)
# Ensure output directories exist (and create them otherwise)
Path("CSVs").mkdir(exist_ok=True)
Path("Images").mkdir(exist_ok=True)

### Helper functions: ###
# Return all qualifying push laps
def get_qualifying_push_laps(session):
    laps = session.laps.copy()
    # Require actual lap times
    laps = laps[laps['LapTime'].notna()]
    # Drop deleted laps if the column exists
    if 'Deleted' in laps.columns:
        laps = laps[~laps['Deleted'].fillna(False)]
    # Keep only accurate laps if the column exists
    if 'IsAccurate' in laps.columns:
        laps = laps[laps['IsAccurate'].fillna(False)]
    # Remove unusually slow laps (5% slower than the fastest)
    fastest = laps['LapTime'].min()
    laps = laps[laps['LapTime'] < fastest * 1.05]
    laps = laps.sort_values('LapTime')
    # # Debugging: Check how many laps are being counted
    # print(
    #     f"{session.event['Location']}: "
    #     f"{len(laps)} qualifying push laps"
    # )
    return laps

# Extract X/Y/Speed/Time telemetry from one lap
def extract_clean_xy_speed(lap):
    pos = lap.get_pos_data()
    car = lap.get_car_data()
    pos = pos[['X', 'Y', 'Time']].dropna()
    car = car[['Speed', 'Time']].dropna()
    if len(pos) < 3 or len(car)<3:
        return None
    # convert x/y from decimeters to meters
    x = pos['X'].to_numpy(dtype=float) * 0.1 # m
    y = pos['Y'].to_numpy(dtype=float) * 0.1 # m
    speed = car['Speed'].to_numpy(dtype=float)  # km/h
    # the time values are not necessarily the same between the pos and car data sources
    t = pos['Time'].dt.total_seconds().to_numpy(dtype=float) # s
    t_speed = car['Time'].dt.total_seconds().to_numpy(dtype=float) # s
    return x, y, speed, t, t_speed

# Needed to average speeds for the colormap onto the same reference positions
def normalized_arclength(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.insert(np.cumsum(ds), 0, 0.0)
    if s[-1] <= 0:
        return None
    # normalized distance traveled around the track (from 0 to 1)
    return s / s[-1] 

# Compute total lap distance from the original X/Y trace and total absolute heading change
def compute_distance_and_turning(x, y):
    # Close the loop
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    # Compute steps between each sample
    dx = np.diff(x)
    dy = np.diff(y)
    # Compute each step distance and the total track distance
    step_distances_m = np.hypot(dx, dy)
    total_distance_km = np.sum(step_distances_m) / 1000
    if total_distance_km <= 0:
            return None, None
    # filter out any steps of size 0 for the heading computation purposes
    # (these repeated spatial points can mess with the headings a lot!)
    keep = np.concatenate(([True], step_distances_m > 0))
    x_clean = x[keep]
    y_clean = y[keep]
    dx = np.diff(x_clean)
    dy = np.diff(y_clean)
    # Compute instantaneous heading vectors (radians)
    headings = np.arctan2(dy, dx)
    # Close the loop with the headings
    headings_closed = np.append(headings, headings[0])
    # Map differences while handling the trig wrap-around (+/- pi)
    delta_headings = np.diff(headings_closed)
    delta_headings = (delta_headings + np.pi) % (2 * np.pi) - np.pi
    delta_degrees = np.abs(np.degrees(delta_headings))
    # Compute cumulative degrees turned
    total_degrees = np.sum(delta_degrees)
    return total_distance_km, total_degrees

# Compute Turning Density for the circuit!
# Loads a specific round and calculates turning density
def process_circuit(year, round_num):
    try:
        # Load the qualifying session
        session = fastf1.get_session(year, round_num, 'Q')
        session.load(telemetry=True, laps=True)
        # Ensure laps are present
        if session.laps is None or session.laps.empty:
                return None
        # Get all valid qualifying laps
        push_laps = get_qualifying_push_laps(session)
        if push_laps.empty:
            return None
        # Use the fastest valid lap as the geometry/reference lap
        fastest_lap = push_laps.iloc[0]
        # Get track event metadata
        circuit_name = session.event['Location']
        # Extract fastest-lap telemetry for the reference X/Y trace
        fastest_data = extract_clean_xy_speed(fastest_lap)
        if fastest_data is None:
            return None
        x, y, speed_raw_fastest, t, t_speed = fastest_data
        # Reference normalized distance coordinate for fastest lap
        s_ref = normalized_arclength(x, y)
        if s_ref is None:
            return None
        # Build averages from all qualifying laps
        speed_profiles_on_ref = []
        distance_values_km = []
        turning_values_deg = []
        # loop through all laps
        for _, lap in push_laps.iterrows():
            # Extract lap data
            lap_data = extract_clean_xy_speed(lap)
            if lap_data is None:
                continue
            x_lap, y_lap, speed_lap, t_lap, t_speed_lap = lap_data
            # Put speed on the position-data time grid
            sort_idx = np.argsort(t_speed_lap)
            speed_on_pos_grid = np.interp(
                t_lap,
                t_speed_lap[sort_idx],
                speed_lap[sort_idx]
            )
            # Distance and total turning for this lap
            lap_distance_km, lap_total_degrees = compute_distance_and_turning(x_lap, y_lap)
            if lap_distance_km is not None and lap_total_degrees is not None:
                distance_values_km.append(lap_distance_km)
                turning_values_deg.append(lap_total_degrees)
            # Normalized arclength for speed interpolation
            s_lap = normalized_arclength(x_lap, y_lap)
            if s_lap is None:
                continue
            # Speed profile interpolation onto fastest-lap reference grid
            s_lap_unique, unique_idx = np.unique(s_lap, return_index=True)
            speed_pos_unique = speed_on_pos_grid[unique_idx]
            if len(s_lap_unique) >= 3:
                speed_interp = np.interp(
                    s_ref,
                    s_lap_unique,
                    speed_pos_unique
                )
                speed_profiles_on_ref.append(speed_interp)
        # Error checks
        if len(speed_profiles_on_ref) == 0:
            return None
        if len(distance_values_km) == 0 or len(turning_values_deg) == 0:
            return None
        # Take the median distance and total turning over all qualifying laps
        # (using median instead of mean because the a small number of outlier laps
        # containing telemetry artifacts can significantly skew total turning)
        distance_km = np.nanmedian(distance_values_km)
        total_degrees = np.nanmedian(turning_values_deg)

        # # Debugging: Printouts that show most circuits have a very small number of 
        # # overpredicting "outlier" map measurements in total degrees turned
        # print(
        #     f"\n{circuit_name}\n"
        #     f"  Distance [km] : "
        #     f"min={np.min(distance_values_km):.3f}, "
        #     f"avg={np.mean(distance_values_km):.3f}, "
        #     f"max={np.max(distance_values_km):.3f}\n"
        #     f"  Turning [deg] : "
        #     f"min={np.min(turning_values_deg):.1f}, "
        #     f"avg={np.mean(turning_values_deg):.1f}, "
        #     f"max={np.max(turning_values_deg):.1f}"
        # )
        # print(
        #     f"Turning percentiles \n"
        #     f"  0%  = {np.percentile(turning_values_deg, 0):.1f}\n"
        #     f"  5%  = {np.percentile(turning_values_deg, 5):.1f}\n"
        #     f" 25%  = {np.percentile(turning_values_deg, 25):.1f}\n"
        #     f" 50%  = {np.percentile(turning_values_deg, 50):.1f}\n"
        #     f" 75%  = {np.percentile(turning_values_deg, 75):.1f}\n"
        #     f" 95%  = {np.percentile(turning_values_deg, 95):.1f}\n"
        #     f"100% = {np.percentile(turning_values_deg, 100):.1f}"
        # )

        # Average speed at every location around the circuit
        speed_avg_quali = np.nanmean(np.vstack(speed_profiles_on_ref), axis=0)
        # Ratio of averaged total turning to averaged lap distance
        turning_density = total_degrees / distance_km

        # # Debugging: Take a look to ensure data is reasonable
        # fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        # axs[0].plot(t, x, color='red', label='Fastest Lap Raw', linewidth=2)
        # axs[0].set_ylabel('X Position [m]')
        # axs[0].grid(True, linestyle='--', alpha=0.5)
        # axs[0].legend(loc='upper right')
        # axs[1].plot(t, y, color='red', label='Fastest Lap Raw', linewidth=2)
        # axs[1].set_ylabel('Y Position [m]')
        # axs[1].grid(True, linestyle='--', alpha=0.5)
        # axs[1].legend(loc='upper right')
        # axs[2].plot(t_speed, speed_raw_fastest, color='red', label='Fastest Lap Raw', linewidth=1.5)
        # axs[2].plot(t, speed_avg_quali, color='blue', label='Average', linestyle=':', linewidth=2)
        # axs[2].set_xlabel('Time [s]')
        # axs[2].set_ylabel('Speed [km/h]')
        # axs[2].grid(True, linestyle='--', alpha=0.5)
        # axs[2].legend(loc='upper right')
        # fig.suptitle(f"Telemetry Analysis: {circuit_name} Qualifying {year} ({fastest_lap['Driver']})")
        # plt.tight_layout()
        # plt.show()

        return {
            'Round': round_num,
            'Circuit': circuit_name,
            'Track Length [km]': distance_km,
            'Total Turning [deg]': total_degrees,
            'Turning Density [deg/km]': turning_density,
            'X Trace': x,
            'Y Trace': y,
            'Speed Trace': speed_avg_quali
        }
    except Exception as e:
        print(f"Skipping Round {round_num} due to error: {e}")
        return None
### End Helper Functions ###

### Compute Results ###
# Loop over a chosen season
all_circuit_data = []
# Fetch total schedule array bounds from fastf1
schedule = fastf1.get_event_schedule(year)
# Filter out pre-season testing events
race_rounds = (
    schedule.loc[
        (schedule['EventFormat'] != 'testing') &
        (schedule['RoundNumber'] > 0),
        'RoundNumber'
    ]
    .astype(int)
    .tolist()
)
# Compute and store data from all circuits
print(f"Beginning analysis on {len(race_rounds)} championship rounds...")
for r_num in race_rounds:
    # Get event metadata from the schedule
    event_row = schedule.loc[schedule['RoundNumber'] == r_num].iloc[0]
    circuit_name = event_row['Location']
    event_name = event_row['EventName']
    event_date = pd.Timestamp(event_row['EventDate']).tz_localize(None)
    # Skip if the event hasn't happened yet
    if event_date > pd.Timestamp.now().tz_localize(None):
        print(
            f"Skipping Round {r_num}: {circuit_name} "
            f"({event_name}) - future event"
        )
        continue
    # Otherwise, compute the results!
    print(f"Analyzing Round {r_num}: {circuit_name} ({event_name})...")
    result = process_circuit(year, r_num)
    if result:
        all_circuit_data.append(result)
    time.sleep(1)  # Polite pause between API calls
# Build, Rank, and Sort the Pandas DataFrame
if not all_circuit_data:
    raise RuntimeError("No circuit data collected.")
df_ranking = pd.DataFrame(all_circuit_data)
df_ranking['Rank'] = df_ranking['Turning Density [deg/km]'].rank(ascending=False, method='first').astype(int)
df_ranking = df_ranking.sort_values(by='Rank').reset_index(drop=True)
# Display results
print(f"\n--- F1 {year} CIRCUITS TURNING DENSITY LEADERBOARD ---")
print(df_ranking[['Rank', 'Circuit', 'Total Turning [deg]', 'Track Length [km]', 'Turning Density [deg/km]']].to_string(index=False))
# Export to CSV file
df_export = df_ranking.drop(columns=['X Trace', 'Y Trace', 'Speed Trace'])
df_export.to_csv(f'CSVs/{year}_F1_turning_density.csv', index=False)



######## VISUALIZE RESULTS ########
# Classify tracks to see the engineering split
# Note: These only currently consider 2025 and 2026 circuits
non_permanent_circuits = [
    # Full street circuits
    'Jeddah',
    'Monte Carlo',
    'Monaco',
    'Baku',
    'Marina Bay',
    'Las Vegas',
    # Temporary/semi-permanent street-style circuits
    'Melbourne',
    'Miami Gardens',
    'Miami',
    'Montréal',
    'Madrid'
]
# Create a classification column
df_ranking['Circuit Type'] = df_ranking['Circuit'].apply(
    lambda x: 'Non-Permanent' if x in non_permanent_circuits else 'Permanent'
)
# Create the horizontal bar plot
sns.set_theme(style="darkgrid")
plt.figure(figsize=(12, 8))
color_palette = {'Non-Permanent': '#ff1801', 'Permanent': '#1f77b4'} # F1 Red vs Sport Blue
ax = sns.barplot(
    x='Turning Density [deg/km]',
    y='Circuit',
    hue='Circuit Type',
    data=df_ranking,
    palette=color_palette,
    dodge=False,
    order=df_ranking['Circuit']
)
# ensure annotations stay within axes limits
ax.set_xlim(0, np.ceil((df_ranking['Turning Density [deg/km]'].max()+50)/50)*50)
# Stylize the chart
plt.title(f'Formula 1 Circuit "Turning Density" Ranking ({year})', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Degrees Turned per Kilometer of Track', fontsize=14, labelpad=10)
plt.ylabel('Grand Prix Circuit', fontsize=14)
plt.legend(title='Circuit Classification', loc='lower right', frameon=True)
# Add data value labels to the end of each bar
for p in ax.patches:
    width = p.get_width()
    if width > 0: # Ensure valid bars are labeled
        ax.text(
            width + df_ranking['Turning Density [deg/km]'].max() * 0.01, # Position slightly past the end of the bar
            p.get_y() + p.get_height() / 2, # Center vertically in the bar row
            f'{width:.1f}°/km',             # Formatted to 1 decimal place
            va='center',
            fontsize=10,
            fontweight='bold'
        )
plt.tight_layout()
# Save the plot
plt.savefig(f'Images/{year}_F1_turning_density_bar_chart.png', dpi=450, bbox_inches="tight")
plt.close()
# plt.show()

######## VISUALIZE LAP XY TRACES COLORED BY SPEED ########
# Number of circuits
n_circuits = len(df_ranking)
# Grid dimensions of the plot
n_cols = 4
n_rows = int(np.ceil(n_circuits / n_cols))
# Generate subplot structure
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4.2 * n_cols, 4.4 * n_rows),
    constrained_layout=True
)
axes = np.array(axes).flatten()
# Global speed limits so the colorbar is consistent across all circuits
vmin = 0
vmax = 400
norm = Normalize(vmin=vmin, vmax=vmax)
# Custom colormap:
speed_cmap = LinearSegmentedColormap.from_list(
    'speed_blue_to_red',
    [
        '#001a66',  # very slow: dark blue
        '#0057ff',  # slow-medium: blue
        '#66ccff',  # medium: light blue
        '#ffb000',  # fast-ish: orange
        '#ff1801',  # fast: red
        '#8a0703'   # fastest: dark red
    ]
)
# Generate each circuit map
for idx, row in df_ranking.iterrows():
    ax = axes[idx]
    ax.set_facecolor('white')
    ax.grid(False)
    x_trace = np.asarray(row['X Trace'], dtype=float)
    y_trace = np.asarray(row['Y Trace'], dtype=float)
    speed_trace = np.asarray(row['Speed Trace'], dtype=float)
    # Center each circuit around its own origin
    x_plot = x_trace - np.nanmean(x_trace)
    y_plot = y_trace - np.nanmean(y_trace)
    # Explicitly close the plotted trace so the start/finish segment is colored
    x_plot_closed = np.append(x_plot, x_plot[0])
    y_plot_closed = np.append(y_plot, y_plot[0])
    speed_trace_closed = np.append(speed_trace, speed_trace[0])
    # Build line segments from consecutive X/Y points, including final point back to start
    points = np.array([x_plot_closed, y_plot_closed]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Use average endpoint speed for each segment
    segment_speeds = 0.5 * (speed_trace_closed[:-1] + speed_trace_closed[1:])
    # Define line collection
    lc = LineCollection(
        segments,
        cmap=speed_cmap,
        norm=norm,
        linewidth=6,
        capstyle='round' # to ensure smooth outer/inner sides of curves
    )
    lc.set_array(segment_speeds)
    ax.add_collection(lc)
    # Set plot limits manually with padding to avoid clipping/overlap
    x_min, x_max = np.nanmin(x_plot), np.nanmax(x_plot)
    y_min, y_max = np.nanmin(y_plot), np.nanmax(y_plot)
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding_fraction = 0.08 # 8% seems to do the trick
    x_padding = padding_fraction * x_range
    y_padding = padding_fraction * y_range
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    # Equal axes to preserve track shape
    ax.set_aspect('equal', adjustable='box')
    # Clean small-multiple look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Annotate
    # Circuit title plus turning density printed
    circuit_name = row['Circuit']
    deg_per_km = row['Turning Density [deg/km]']
    ax.set_title(
        f"{row['Rank']}. {circuit_name}\n{deg_per_km:.1f}°/km",
        fontsize=26,
        fontweight='bold'
    )
# Hide unused subplot axes
for empty_idx in range(n_circuits, len(axes)):
    axes[empty_idx].axis('off')
# Overall title
fig.suptitle(
    f'Formula 1 Circuit "Turning Density" Ranking ({year})',
    fontsize=52,
    fontweight='bold'
)
# Shared colorbar
sm = ScalarMappable(norm=norm, cmap=speed_cmap)
sm.set_array([])
cbar = fig.colorbar(
    sm,
    ax=axes[:n_circuits],
    orientation='horizontal',
    fraction=0.05,
    pad=0.05
)
cbar.set_label('Speed (km/h)', fontsize=36)
# Format colorbar ticks
speed_ticks = np.linspace(vmin, vmax, 9)
cbar.set_ticks(speed_ticks)
cbar.set_ticklabels([f'{tick:.0f}' for tick in speed_ticks])
cbar.ax.tick_params(labelsize=26)
# Save and show
plt.savefig(
    f'Images/{year}_F1_turning_density_circuit_maps.png',
    dpi=450,
    bbox_inches='tight'
)
plt.close()
# plt.show()