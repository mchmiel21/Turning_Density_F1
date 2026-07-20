# Turning_Density_F1
Analysis and visualizations of Formula 1 Circuit ***Turning Densities***

## Description
The proposed ***Turning Density (TD)*** metric is defined as the cumulative absolute change in vehicle path heading normalized by lap distance. Therefore, higher values indicate more directionally complex circuits while lower values indicate straighter circuits.

The ***Lateral G-Weighted Turning Density (LGWTD)*** metric weights *Turning Density* by how many Lateral Gs are being experienced during each turn (each discrete sample of heading change). Therefore, high-G turns, especially sustained high-G turns, will contribute more heavily than low-G turns, in an attempt to better characterize the "severity" of a circuit's turning profile.

<!-- View analysis related to these results here: [ANALYSIS.md](ANALYSIS.md) -->

- `compute_TD_and_LGWTD.py` calculates and stores *Turning Density* and *Lateral G-Weighted Turning Density* results for Formula 1 circuits of any given year, based on telemetry data from all push/flying laps from qualifying. 
    - The only required user input is the calendar year/season you would like to run (the `year` variable at the very top of the script).
- `make_plots.py` generates and saves plots to the "Result Images" directory, using the data previously computed by `compute_TD_and_LGWTD.py`.
    - Types of output plots are described in the following **Outputs** section.
    - Each function (called in the main function) creates and saves a different type of plot.

## Outputs
- Raw data:
    - CSV files containing the computed *Turning Density* (in deg/km) and *Lateral G-Weighted Turning Density* (in G-deg/km) and corresponding ranks, as well as the total turning (in degrees) and total distance (in km) used to calculate *Turning Density*
    - Parquet files containing all data in the CSV files, plus X/Y coordinate traces of each circuit and average speeds and lateral G forces experienced at all X/Y coordinates from qualifying
- Plots/Images:
    - Bar charts ranking all circuits by *TD*, color-coded by circuit type (permanent vs. non-permanent)
    - Bar charts ranking all circuits by *LGWTD*, color-coded by track conditions (dry vs. wet)
    - Bar charts with multiple bars to directly compare *TD* and/or *LGWTD* between different years at the same track
        - Color-coding is the same as the single bar charts
        - Only circuits contained in all years considered are included
        - Ordered by the maximum metric value of each circuit across all years considered
    - A pictorial grid of circuit maps colored by qualifying speed, ordered from top-left to bottom-right by *TD* (highest to lowest)
    - A pictorial grid of circuit maps colored by lateral G-force, ordered from top-left to bottom-right by *LGWTD* (highest to lowest)
        - Any wet condition qualifying sessions are noted as "(Wet Track)", as these values will be significantly lower than they otherwise would have been on a dry track

## Methodology
- This code uses data from all valid qualifying push laps as follows: 
    1. Total *Turning Density* and *Lateral G-Weighted Turning Density* values were calculated using interquartile means computed over all valid push laps. The IQM was used rather than the mean because a small number of laps on some circuits contained telemetry artifacts which inflated total turning values.
    2. Speed traces and Lateral G traces are averaged to get accurate profiles throughout the lap, robust against data dropouts from a handful of cars' 'Speed' data.
    3. Laps more than 5% slower than the fastest lap of qualifying were ignored.

## Limitations
- *Turning Density* is computed from [Fast-F1](https://github.com/theOehrly/Fast-F1) X/Y position channels, which are local track-position coordinates. However, as [theOehrly stated](https://github.com/theOehrly/Fast-F1/discussions/116), these locations are "normalized track positions", which "tend to approximately follow the ideal racing line around a circuit". Unfortunately, they are not actual GPS racing lines, but they seem to be the best data available. 
    - Small variations (+/-0.5%) in *TD* exist from year-to-year, based on the telemetry X/Y position data mappings.
- The Lateral Gs used to compute *Lateral G-Weighted Turning Density* are derived using functions from [@TracingInsights](https://github.com/TracingInsights), which are based on the turn rate (computed from the X/Y position data) in conjunction with Speed data.
    - The metric's utility as a comparison between circuits is **heavily** limited by the following:
        - Since bank angle data is not available in Fast-F1, the Lateral G force computed by this code on highly banked corners will be overpredicted.
        - As discussed on the [Fast-F1 issues page](https://github.com/theOehrly/Fast-F1/issues/507), the data available via Fast-F1 is not at a high enough frequency or precision to reliably compute accelerations. The approximations used in this *LGWTD* metric (from @TracingInsights functions) seem to be reasonable, but are certainly not totally accurate. 
        - Since *LGWTD* depends on the speed of the car, it is not a purely circuit-geometry-based metric. Thus, it will yield differing results from year to year (with different car regulations) or based on the conditions of each particular qualifying session (wet vs. dry). It might still be interesting to look at, though, or to compare races versus each other, if not simply the circuits themselves!

## Requirements
This script requires Python along with:
- fastf1
- numpy
- pandas
    - pyarrow
- matplotlib
- seaborn

## Acknowledgments
- All Formula 1 telemetry data is obtained via the [Fast-F1 python package](https://github.com/theOehrly/Fast-F1).
- The Lateral Gs used to compute *Lateral G-Weighted Turning Density* are derived using two functions that were directly copied from [@TracingInsights](https://github.com/TracingInsights) to the `utils_TracingInsights.py` file.

## Example Results from 2025:

![2025 Turning Density Circuit Maps Plot](<Result Images/Circuit Maps/2025_F1_TD_circuit_maps.png>)

![2025 LGWTD Circuit Maps Plot](<Result Images/Circuit Maps/2025_F1_LGWTD_circuit_maps.png>)

![2025 Turning Density Bar Chart](<Result Images/Bar Charts/2025_F1_TD_bar_chart.png>)

![2025 LGWTD Bar Chart](<Result Images/Bar Charts/2025_F1_LGWTD_bar_chart.png>)

![2024-2026 LGWTD Bar Chart](<Result Images/Bar Charts/2024-2025-2026_F1_LGWTD_bar_chart.png>)

## Notice
Turning_Density_F1 and this website are unofficial and are not associated in any way with the Formula 1 companies. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks are trade marks of Formula One Licensing B.V.
