"""
Functions copied from Tracing Insights to compute accelerations experienced by the car:
https://github.com/TracingInsights/2026/blob/main/Q.py
(accessed on 14 July 2026)

@author: TracingInsights

Utilized under TracingInsights' Apache License 2.0 (copied to this repository as "LICENSE_TracingInsights")
"""

import numpy as np
import pandas as pd
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

EPS = np.finfo(float).eps
# Pre-allocated smoothing kernels
_KERNEL_3 = np.ones(3, dtype=np.float64) / 3.0
_KERNEL_9 = np.ones(9, dtype=np.float64) / 9.0

# ---------------------------------------------------------------------------
# Helper Functions (Copied from main_optimized.py for standalone execution)
# ---------------------------------------------------------------------------

def _smooth_outliers(arr: np.ndarray, threshold: float, use_abs: bool) -> None:
    if use_abs:
        mask = np.abs(arr) > threshold
    else:
        mask = arr > threshold
    if mask.any():
        indices = np.where(mask)[0]
        indices = indices[(indices >= 1) & (indices < len(arr) - 1)]
        if len(indices) > 0:
            arr[indices] = arr[indices - 1]
            
def _compute_accelerations(
    speed: np.ndarray,
    time_arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Convert speed km/h -> m/s as float64
    vx = speed * (1.0 / 3.6)
    if vx.dtype != np.float64:
        vx = vx.astype(np.float64)
    time_f = (time_arr / np.timedelta64(1, "s")).astype(np.float64)

    # Ensure float64 only when needed
    x_f = x if x.dtype == np.float64 else x.astype(np.float64)
    y_f = y if y.dtype == np.float64 else y.astype(np.float64)
    z_f = z if z.dtype == np.float64 else z.astype(np.float64)
    dist_f = dist if dist.dtype == np.float64 else dist.astype(np.float64)

    # --- X acceleration ---
    dtime = np.gradient(time_f)
    ax = np.gradient(vx) / dtime
    _smooth_outliers(ax, 25.0, use_abs=False)
    ax = np.convolve(ax, _KERNEL_3, mode="same")

    # --- Shared gradient for Y and Z ---
    dx = np.gradient(x_f)
    ds = np.gradient(dist_f)

    # --- Y acceleration ---
    dy = np.gradient(y_f)
    theta = np.arctan2(dy, dx + EPS)
    theta[0] = theta[1]
    dtheta = np.gradient(np.unwrap(theta))
    _smooth_outliers(dtheta, 0.5, use_abs=True)
    C = dtheta / (ds + 0.0001)
    ay = np.square(vx) * C
    ay[np.abs(ay) > 150] = 0
    ay = np.convolve(ay, _KERNEL_9, mode="same")

    # --- Z acceleration ---
    dz = np.gradient(z_f)
    z_theta = np.arctan2(dz, dx + EPS)
    z_theta[0] = z_theta[1]
    z_dtheta = np.gradient(np.unwrap(z_theta))
    _smooth_outliers(z_dtheta, 0.5, use_abs=True)
    z_C = z_dtheta / (ds + 0.0001)
    az = np.square(vx) * z_C
    az[np.abs(az) > 150] = 0
    az = np.convolve(az, _KERNEL_9, mode="same")

    return ax, ay, az, time_f