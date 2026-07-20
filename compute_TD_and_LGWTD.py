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
Path("Result DataFrame Parquets").mkdir(exist_ok=True)

###### Helper functions: ######
# Return all qualifying push laps and whether the session was run on 
# 33% or more inters or wet tires
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

    # Determine wet or dry conditions based on tires
    wet_or_dry = "Dry" # default
    if not laps.empty and 'Compound' in laps.columns:
        # Cast to string and uppercase to protect against variations
        compounds = laps['Compound'].astype(str).str.upper()
        
        # Count rows where tire is INTERMEDIATE or WET
        wet_lap_count = compounds.isin(['INTERMEDIATE', 'WET']).sum()
        total_lap_count = len(laps)
        
        # Determine threshold (1/3rd of laps)
        if (wet_lap_count / total_lap_count) >= (1/3):
            wet_or_dry = "Wet"
        else:
            wet_or_dry = "Dry"
    return laps, wet_or_dry

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
        push_laps, wet_or_dry = get_qualifying_push_laps(session)
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
        average_speeds = []
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
            x_lap, y_lap, _, t_lap, speed_lap, dist_lap = lap_data
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
                average_speeds.append((dist_lap[-1]/1000)/(t_lap[-1]/3600)) # km/h
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
        avg_speed_avg = np.nanmean(average_speeds)

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
            'Track Condition': wet_or_dry,
            'Track Length [km]': distance_km,
            'Average Speed [km/h]': avg_speed_avg,
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
if __name__ == "__main__":
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
    # Export to CSV file
    turning_density_df_ranking = pd.DataFrame(all_circuit_data)
    df_export = turning_density_df_ranking.drop(columns=['X Trace', 'Y Trace', 'Speed Trace', 'Lateral G-Force Trace'])
    df_export.to_csv(f'Result CSVs/{year}_F1_TD_and_LGWTD.csv', index=False)
    # Save full DataFrame to Parquet file
    turning_density_df_ranking.to_parquet(f'Result DataFrame Parquets/{year}_F1_TD_and_LGWTD.parquet', index=False)
    # Rank and print by Turning Density
    turning_density_df_ranking['Rank'] = turning_density_df_ranking['Turning Density [deg/km]'].rank(ascending=False, method='first').astype(int)
    turning_density_df_ranking = turning_density_df_ranking.sort_values(by='Rank').reset_index(drop=True)
    # Display results
    print(f"\n--------------- F1 {year} CIRCUITS TURNING DENSITY LEADERBOARD ---------------")
    print(turning_density_df_ranking[['Rank', 'Circuit', 'Track Condition', 'Total Turning [deg]', 'Track Length [km]', 'Average Speed [km/h]', 'Turning Density [deg/km]', 'Lateral G-Weighted Turning Density [G-deg/km]']].to_string(index=False))
    # Rank and print by Lateral G-Weighted Turning Density
    latG_weighted_turning_density_df_ranking = pd.DataFrame(all_circuit_data)
    latG_weighted_turning_density_df_ranking['Rank'] = latG_weighted_turning_density_df_ranking['Lateral G-Weighted Turning Density [G-deg/km]'].rank(ascending=False, method='first').astype(int)
    latG_weighted_turning_density_df_ranking = latG_weighted_turning_density_df_ranking.sort_values(by='Rank').reset_index(drop=True)
    # Display results
    print(f"\n--------------- F1 {year} CIRCUITS LATERAL G-WEIGHTED TURNING DENSITY LEADERBOARD ---------------")
    print(latG_weighted_turning_density_df_ranking[['Rank', 'Circuit', 'Track Condition', 'Total Turning [deg]', 'Track Length [km]', 'Average Speed [km/h]', 'Turning Density [deg/km]', 'Lateral G-Weighted Turning Density [G-deg/km]']].to_string(index=False))