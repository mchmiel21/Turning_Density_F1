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
# import function from @TracingInsights to compute car accelerations
from from_TracingInsights.utils_TracingInsights import _compute_accelerations

######## COMPUTE TURNING DENSITY METRICS! ########
# Enable caching (so you don't re-download data each time)
CACHE_DIR = Path("f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))
# Suppress terminal logging
fastf1.set_log_level(logging.ERROR)
# Ensure output directories exist (and create them otherwise)
Path("Result CSVs").mkdir(exist_ok=True)
Path("Result Images").mkdir(exist_ok=True)
Path(f"Result Images/Bar Charts").mkdir(exist_ok=True)
Path(f"Result Images/Circuit Maps").mkdir(exist_ok=True)

###### Helper functions: ######
# Return all qualifying push laps
def get_qualifying_push_laps(session):
    laps = session.laps.copy()
    # Require actual lap times
    laps = laps[laps['LapTime'].notna()]
    # Drop deleted laps if the column exists
    if 'Deleted' in laps.columns:
        laps = laps[~laps['Deleted'].astype(bool).fillna(False)]
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

def drop_repeat_xy_points(x, y, z, t, speed, dist):
    # Compute steps between each sample
    dx = np.diff(x)
    dy = np.diff(y)
    # Compute each step distance
    step_distances_m = np.hypot(dx, dy)
    # filter out any steps of size 0 m (these repeated spatial points can mess with the headings a lot!)
    keep = np.concatenate(([True], step_distances_m > 0))
    x_clean = x[keep]
    y_clean = y[keep]
    z_clean = z[keep]
    t_clean = t[keep]
    speed_clean = speed[keep]
    dist_clean = dist[keep]

    return x_clean, y_clean, z_clean, t_clean, speed_clean, dist_clean

# Get total distance traveled based on X/Y/Z coordinates
def get_distance(x, y, z):
    # There are sometimes issues with the distance directly from Fast-F1 
    # (it's computed by integrating speed data, so if speed data is bad, it's wrong)
    # (example: Kimi's pole lap in Q3 of Suzuka 2026)
    # So, compute it using x and y instead for robustness
    dx = np.diff(x) # m
    dy = np.diff(y) # m
    dz = np.diff(z) # m
    step_distances = np.sqrt(dx**2 + dy**2 + dz**2) # m
    dist = np.insert(np.cumsum(step_distances), 0, 0.0)

    return dist

# Extract X/Y/Speed/Time/Distance telemetry from one lap
def extract_clean_data(lap, lap_tel_data):
    pos = lap.get_pos_data()
    tel = lap_tel_data
    pos = pos[['X', 'Y', 'Z', 'Time']].dropna()
    tel = tel[['Speed', 'Time']].dropna()
    if len(pos) < 3 or len(tel)<3:
        return None
    # convert x/y from decimeters to meters
    x = pos['X'].to_numpy(dtype=float) * 0.1 # m
    y = pos['Y'].to_numpy(dtype=float) * 0.1 # m
    z = pos['Z'].to_numpy(dtype=float) * 0.1 # m
    speed = tel['Speed'].to_numpy(dtype=float)  # km/h
    # the time values are not necessarily the same between the pos and tel data sources... must interpolate onto same grid
    t = pos['Time'].dt.total_seconds().to_numpy(dtype=float) # s
    t_tel = tel['Time'].dt.total_seconds().to_numpy(dtype=float) # s
    speed = np.interp(t, t_tel, speed) # km/h
    dist = get_distance(x, y, z) # m
    # drop repeated combos of x and y:
    x, y, z, t, speed, dist = drop_repeat_xy_points(x, y, z, t, speed, dist)

    return x, y, z, t, speed, dist

# Compute variables related to Turning Density for a single lap
def compute_turning_density_values(x, y):
    # # Compute steps between each sample
    dx = np.diff(x)
    dy = np.diff(y)
    if len(dx) == 0:
        return None, None
    # compute turning density
    # Compute instantaneous heading vectors (radians)
    headings = np.arctan2(dy, dx)
    # append last heading to match the length of inputs
    headings = np.append(headings, headings[-1]) 
    # Map differences while handling the trig wrap-around (+/- pi)
    headings = np.unwrap(headings) 
    # Compute the differences in headings between samples
    # (i.e. how much the car has turned)
    delta_headings = np.diff(headings)
    # both left and right turns count positively
    delta_degrees = np.abs(np.degrees(delta_headings))
    # make same length as the input arrays
    delta_degrees = np.concatenate(([0.0], delta_degrees))
    # sum to find total degrees turned
    total_degrees_turned_lap = np.sum(delta_degrees)

    return delta_degrees, total_degrees_turned_lap

# Compute Turning Density for the circuit!
# Loads a specific round and calculates turning density
def process_circuit(year, round_num):
    try:
        # Load the qualifying session
        session = fastf1.get_session(year, round_num, 'Q')
        session.load(telemetry=True, laps=True, weather = False, messages = False)
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
        telemetry_fastest = fastest_lap.get_telemetry()
        fastest_data = extract_clean_data(fastest_lap, telemetry_fastest)
        if fastest_data is None:
            return None
        x, y, z, _, speed_fastest, dist_fastest = fastest_data
        # Reference normalized distance coordinate for fastest lap (used for plotting circuits)
        s_ref = dist_fastest / dist_fastest[-1] 
        if s_ref is None:
            return None
        # Build averages from all qualifying laps
        distance_values_km = []
        turning_values_deg = [] 
        g_turn_sum_values = []  
        speed_profiles_on_ref = []
        latG_profiles_on_ref = []
        # loop through all laps
        for _, lap in push_laps.iterrows():
            # get the lap data
            try:
                telemetry = lap.get_telemetry() 
            except ValueError as e:
                # If the lap data is broken, print a warning and skip it
                print(f"Skipping corrupted lap {lap['LapNumber']} for driver {lap['Driver']}: {e}")
                continue
            lap_data = extract_clean_data(lap, telemetry)
            if lap_data is None:
                continue
            x_lap, y_lap, _, _, speed_lap, dist_lap = lap_data
            # find Turning Density for this lap
            delta_degrees_lap, total_degrees_turned_lap = compute_turning_density_values(x_lap, y_lap)
            if delta_degrees_lap is None:
                continue
            # Find G-Weighted Turning Density for this lap using the Tracing Insights acceleration computation functions
            # Requires using pure get_telemetry data, rather than get_pos_data, because the latter is too sparse to compute accurate accelerations
            time_arr_TI = telemetry["Time"].to_numpy()
            speed_TI = telemetry["Speed"].to_numpy()
            x_TI = telemetry["X"].to_numpy()
            y_TI = telemetry["Y"].to_numpy()
            z_TI = telemetry["Z"].to_numpy()
            dist_TI = get_distance(x_TI*0.1, y_TI*0.1, z_TI*0.1) # convert inputs to meters first
            # Call TurningInsights accelerations function
            _, ay_lap, _, _ = _compute_accelerations(speed_TI, time_arr_TI, x_TI, y_TI, z_TI, dist_TI)
            # interpolate the lateral Gs onto the same reference grid as the turning density values
            ay_lap = np.interp(dist_lap, dist_TI, ay_lap)
            latGs_lap = np.abs(ay_lap) / 9.80665 # lateral Gs (magnitude only)
            total_g_times_delta_degrees_lap = np.sum(np.abs(latGs_lap) * delta_degrees_lap) # G-deg
            # Assign lap values to total circuit trackers (if nothing is missing)
            if dist_lap is not None and total_degrees_turned_lap is not None and ay_lap is not None:
                distance_values_km.append(dist_lap[-1]/1000) # km
                turning_values_deg.append(total_degrees_turned_lap) # deg
                g_turn_sum_values.append(total_g_times_delta_degrees_lap) # G-deg
            # interpolate Speed profile and Lateral G profile interpolation onto fastest-lap reference grid
            # (for coloring circuit maps)
            s_interp = dist_lap / dist_lap[-1] # normalized distance traveled around the track (from 0 to 1)
            speed_profiles_on_ref.append(
                np.interp(s_ref, s_interp, speed_lap)
            )
            latG_profiles_on_ref.append(
                np.interp(s_ref, s_interp, latGs_lap)
            )

        # Error checks
        if len(speed_profiles_on_ref) == 0 or len(latG_profiles_on_ref) == 0:
            return None
        if len(distance_values_km) == 0 or len(turning_values_deg) == 0:
            return None
        # Take the Interquartile Means of results over all qualifying laps
        # (using Interquartile Mean instead of mean because the a small number of outlier laps
        # containing telemetry artifacts can significantly skew total turning)
        dist_arr = np.asarray(distance_values_km)
        turn_arr = np.asarray(turning_values_deg)
        g_turn_arr = np.asarray(g_turn_sum_values)
        # Compute the interquartile mean for distance and total turning
        q25_dist = np.nanpercentile(dist_arr, 25)
        q75_dist = np.nanpercentile(dist_arr, 75)
        mask_dist = (dist_arr >= q25_dist) & (dist_arr <= q75_dist)
        distance_km = np.nanmean(dist_arr[mask_dist])
        # Compute the interquartile mean for total turning
        q25_turn = np.nanpercentile(turn_arr, 25)
        q75_turn = np.nanpercentile(turn_arr, 75)
        mask_turn = (turn_arr >= q25_turn) & (turn_arr <= q75_turn)
        total_degrees = np.nanmean(turn_arr[mask_turn])
        # compute interquartile mean for G times degrees turned
        q25_gturn = np.nanpercentile(g_turn_arr, 25)
        q75_gturn = np.nanpercentile(g_turn_arr, 75)
        mask_gturn = (g_turn_arr >= q25_gturn) & (g_turn_arr <= q75_gturn)
        total_g_times_delta_degrees = np.nanmean(g_turn_arr[mask_gturn])

        # Compute TD and LGWTD metrics!
        turning_density = total_degrees/distance_km
        g_weighted_turning_density = total_g_times_delta_degrees/distance_km

        # Average speed and lateral G-force at every location around the circuit 
        # (for color-coding circuit maps)
        speed_avg_quali = np.nanmean(np.vstack(speed_profiles_on_ref), axis=0)
        latG_avg_quali = np.nanmean(np.vstack(latG_profiles_on_ref), axis=0)

        # # Debugging: Take a look to ensure data is reasonable
        # print(f"Turning Density = {turning_density:.2f} deg/km")
        # print(f"G-Weighted Turning Density = {g_weighted_turning_density:.2f} G-deg/km")
        # fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        # axs[0].plot(dist_fastest, x, color='red', label='Fastest Lap Raw', linewidth=2)
        # axs[0].set_ylabel('X Position [m]')
        # axs[0].grid(True, linestyle='--', alpha=0.5)
        # axs[0].legend(loc='upper right')
        # axs[1].plot(dist_fastest, y, color='red', label='Fastest Lap Raw', linewidth=2)
        # axs[1].set_ylabel('Y Position [m]')
        # axs[1].grid(True, linestyle='--', alpha=0.5)
        # axs[1].legend(loc='upper right')
        # axs[2].plot(dist_fastest, speed_fastest, color='red', label='Fastest Lap Raw', linewidth=1.5)
        # axs[2].plot(dist_fastest, speed_avg_quali, color='blue', label='Average', linestyle=':', linewidth=2)
        # axs[2].set_ylabel('Speed [km/h]')
        # axs[2].grid(True, linestyle='--', alpha=0.5)
        # axs[2].legend(loc='upper right')# axs[2].plot(t_speed, speed_raw_fastest, color='red', label='Fastest Lap Raw', linewidth=1.5)
        # axs[3].plot(dist_fastest, latG_avg_quali, color='blue', label='Average', linewidth=2)
        # axs[3].set_xlabel('Distance [m]')
        # axs[3].set_ylabel('Lateral G-Force')
        # axs[3].grid(True, linestyle='--', alpha=0.5)
        # axs[3].legend(loc='upper right')
        # fig.suptitle(f"Telemetry Analysis: {circuit_name} Qualifying {year} ({fastest_lap['Driver']})")
        # plt.tight_layout()
        # plt.show()

        return {
            'Round': round_num,
            'Circuit': circuit_name,
            'Track Length [km]': distance_km,
            'Total Turning [deg]': total_degrees,
            'Turning Density [deg/km]': turning_density,
            'Lateral G-Weighted Turning Density [G-deg/km]': g_weighted_turning_density,
            'X Trace': x,
            'Y Trace': y,
            'Speed Trace': speed_avg_quali,
            'Lateral G-Force Trace': latG_avg_quali
        }
    except Exception as e:
        print(f"Skipping Round {round_num} due to error: {e}")
        return None
###### End Helper Functions ######

###### Compute Results ######
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
    # Skip if the qualifying session hasn't happened yet
    if event_date > pd.Timestamp.now().tz_localize(None) + pd.Timedelta(hours=24):
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
# Rank and print by Turning Density
turning_density_df_ranking = pd.DataFrame(all_circuit_data)
turning_density_df_ranking['Rank'] = turning_density_df_ranking['Turning Density [deg/km]'].rank(ascending=False, method='first').astype(int)
turning_density_df_ranking = turning_density_df_ranking.sort_values(by='Rank').reset_index(drop=True)
# Display results
print(f"\n--- F1 {year} CIRCUITS TURNING DENSITY LEADERBOARD ---")
print(turning_density_df_ranking[['Rank', 'Circuit', 'Total Turning [deg]', 'Track Length [km]', 'Turning Density [deg/km]', 'Lateral G-Weighted Turning Density [G-deg/km]']].to_string(index=False))
# Export to CSV file
df_export = turning_density_df_ranking.drop(columns=['X Trace', 'Y Trace', 'Speed Trace', 'Lateral G-Force Trace'])
df_export.to_csv(f'Result CSVs/{year}_F1_turning_density.csv', index=False)
# Rank and print by Lateral G-Weighted Turning Density
latG_weighted_turning_density_df_ranking = pd.DataFrame(all_circuit_data)
latG_weighted_turning_density_df_ranking['Rank'] = latG_weighted_turning_density_df_ranking['Lateral G-Weighted Turning Density [G-deg/km]'].rank(ascending=False, method='first').astype(int)
latG_weighted_turning_density_df_ranking = latG_weighted_turning_density_df_ranking.sort_values(by='Rank').reset_index(drop=True)
# Display results
print(f"\n--- F1 {year} CIRCUITS LATERAL G-WEIGHTED TURNING DENSITY LEADERBOARD ---")
print(latG_weighted_turning_density_df_ranking[['Rank', 'Circuit', 'Total Turning [deg]', 'Track Length [km]', 'Turning Density [deg/km]', 'Lateral G-Weighted Turning Density [G-deg/km]']].to_string(index=False))
# Export to CSV file
df_export = latG_weighted_turning_density_df_ranking.drop(columns=['X Trace', 'Y Trace', 'Speed Trace', 'Lateral G-Force Trace'])
df_export.to_csv(f'Result CSVs/{year}_F1_LGWTD.csv', index=False)


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
turning_density_df_ranking['Circuit Type'] = turning_density_df_ranking['Circuit'].apply(
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
    data=turning_density_df_ranking,
    palette=color_palette,
    dodge=False,
    order=turning_density_df_ranking['Circuit']
)
# ensure annotations stay within axes limits
ax.set_xlim(0, np.ceil((turning_density_df_ranking['Turning Density [deg/km]'].max()+50)/50)*50)
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
            width + turning_density_df_ranking['Turning Density [deg/km]'].max() * 0.01, # Position slightly past the end of the bar
            p.get_y() + p.get_height() / 2, # Center vertically in the bar row
            f'{width:.1f}°/km',         
            va='center',
            fontsize=10,
            fontweight='bold'
        )
plt.tight_layout()
# Save the plot
plt.savefig(f'Result Images/Bar Charts/{year}_F1_TD_bar_chart.png', dpi=450, bbox_inches="tight")
plt.close()
# plt.show()

# Create a classification column
latG_weighted_turning_density_df_ranking['Circuit Type'] = latG_weighted_turning_density_df_ranking['Circuit'].apply(
    lambda x: 'Non-Permanent' if x in non_permanent_circuits else 'Permanent'
)
# Create the horizontal bar plot
sns.set_theme(style="darkgrid")
plt.figure(figsize=(12, 8))
color_palette = {'Non-Permanent': '#ff1801', 'Permanent': '#1f77b4'} # F1 Red vs Sport Blue
ax = sns.barplot(
    x='Lateral G-Weighted Turning Density [G-deg/km]',
    y='Circuit',
    hue='Circuit Type',
    data=latG_weighted_turning_density_df_ranking,
    palette=color_palette,
    dodge=False,
    order=latG_weighted_turning_density_df_ranking['Circuit']
)
# ensure annotations stay within axes limits
ax.set_xlim(0, np.ceil((turning_density_df_ranking['Lateral G-Weighted Turning Density [G-deg/km]'].max()+250)/250)*250)
# Stylize the chart
plt.title(f'Formula 1 Circuit "Lateral G-Weighted Turning Density" Ranking ({year})', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Lateral Gs times Degrees Turned per Kilometer of Track', fontsize=14, labelpad=10)
plt.ylabel('Grand Prix Circuit', fontsize=14)
plt.legend(title='Circuit Classification', loc='lower right', frameon=True)
# Add data value labels to the end of each bar
for p in ax.patches:
    width = p.get_width()
    if width > 0: # Ensure valid bars are labeled
        ax.text(
            width + latG_weighted_turning_density_df_ranking['Lateral G-Weighted Turning Density [G-deg/km]'].max() * 0.01, # Position slightly past the end of the bar
            p.get_y() + p.get_height() / 2, # Center vertically in the bar row
            f'{width:.1f} G-°/km',    
            va='center',
            fontsize=10,
            fontweight='bold'
        )
plt.tight_layout()
# Save the plot
plt.savefig(f'Result Images/Bar Charts/{year}_F1_LGWTD_bar_chart.png', dpi=450, bbox_inches="tight")
plt.close()
# plt.show()

######## VISUALIZE LAP XY TRACES COLORED BY SPEED ########
# Number of circuits
n_circuits = len(turning_density_df_ranking)
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
vmin = 50
vmax = 400
norm = Normalize(vmin=vmin, vmax=vmax)
# speed_cmap = plt.colormaps['turbo']
# Custom colormap:
speed_cmap = LinearSegmentedColormap.from_list(
    'speed_cmap',
    [
        "#001041",  # very slow: dark blue
        '#0057ff',  # slow-medium: blue
        '#66ccff',  # medium: light blue
        '#ffb000',  # fast-ish: orange
        "#b90f00",  # fast: red
        "#690501"   # fastest: dark red
    ]
)
# Generate each circuit map
for idx, row in turning_density_df_ranking.iterrows():
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
# Format the colorbar
cbar = fig.colorbar(
    sm,
    ax=axes[:n_circuits],
    orientation='horizontal',
    fraction=0.05,
    pad=0.05
)
cbar.set_label('Speed [km/h]', fontsize=36)
# Format colorbar ticks
speed_ticks = np.linspace(vmin, vmax, 8)
cbar.set_ticks(speed_ticks)
cbar.set_ticklabels([f'{tick:.0f}' for tick in speed_ticks])
cbar.ax.tick_params(labelsize=26)
# Save and show
plt.savefig(
    f'Result Images/Circuit Maps/{year}_F1_TD_circuit_maps.png',
    dpi=450,
    bbox_inches='tight'
)
plt.close()
# plt.show()

######## VISUALIZE LAP XY TRACES COLORED BY LATERAL Gs ########
# Number of circuits
n_circuits = len(latG_weighted_turning_density_df_ranking)
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
Gmin = 0
Gmax = 6
norm = Normalize(vmin=Gmin, vmax=Gmax)
latG_cmap = plt.colormaps['afmhot'] # 'hot' might be better? or go 0-6.5?
# Generate each circuit map
for idx, row in latG_weighted_turning_density_df_ranking.iterrows():
    ax = axes[idx]
    ax.set_facecolor('white')
    ax.grid(False)
    x_trace = np.asarray(row['X Trace'], dtype=float)
    y_trace = np.asarray(row['Y Trace'], dtype=float)
    latG_trace = np.asarray(row['Lateral G-Force Trace'], dtype=float)
    # Center each circuit around its own origin
    x_plot = x_trace - np.nanmean(x_trace)
    y_plot = y_trace - np.nanmean(y_trace)
    # Explicitly close the plotted trace so the start/finish segment is colored
    x_plot_closed = np.append(x_plot, x_plot[0])
    y_plot_closed = np.append(y_plot, y_plot[0])
    latG_trace_closed = np.append(latG_trace, latG_trace[0])
    # Build line segments from consecutive X/Y points, including final point back to start
    points = np.array([x_plot_closed, y_plot_closed]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Use average endpoint speed for each segment
    segment_latG = 0.5 * (latG_trace_closed[:-1] + latG_trace_closed[1:])
    # Define line collection
    lc = LineCollection(
        segments,
        cmap=latG_cmap,
        norm=norm,
        linewidth=6,
        capstyle='round' # to ensure smooth outer/inner sides of curves
    )
    lc.set_array(segment_latG)
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
    Gdeg_per_km = row['Lateral G-Weighted Turning Density [G-deg/km]']
    ax.set_title(
        f"{row['Rank']}. {circuit_name}\n{Gdeg_per_km:.1f} G-°/km",
        fontsize=26,
        fontweight='bold'
    )
# Hide unused subplot axes
for empty_idx in range(n_circuits, len(axes)):
    axes[empty_idx].axis('off')
# Overall title
fig.suptitle(
    f'Formula 1 Circuit "Lateral G-Weighted Turning Density" Ranking ({year})',
    fontsize=40,
    fontweight='bold'
)
# Shared colorbar
sm = ScalarMappable(norm=norm, cmap=latG_cmap)
sm.set_array([])
cbar = fig.colorbar(
    sm,
    ax=axes[:n_circuits],
    orientation='horizontal',
    fraction=0.05,
    pad=0.05
)
cbar.set_label('Lateral Gs', fontsize=36)
# Format colorbar ticks
latG_ticks = np.linspace(Gmin, Gmax, Gmax+1)
cbar.set_ticks(latG_ticks)
cbar.set_ticklabels([f'{tick:.0f}' for tick in latG_ticks])
cbar.ax.tick_params(labelsize=26)
# Save and show
plt.savefig(
    f'Result Images/Circuit Maps/{year}_F1_LGWTD_circuit_maps.png',
    dpi=450,
    bbox_inches='tight'
)
plt.close()
# plt.show()