
"""
Section II vehicul going up or down a 16m pipe 

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------[ CONFIG ]--------------------
VEHICLE = 2  # 1 or 2

G = 9.805  
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(ROOT, "projectfiles")
PLOTS_DIR = os.path.join(ROOT, "plot_Section_II")
os.makedirs(PLOTS_DIR, exist_ok=True)
SEC1_ACC = os.path.join(PROJECT, "secI_acc.csv")
SEC1_GYR = os.path.join(PROJECT, "secI_gyr.csv")
ACC_FILE = os.path.join(PROJECT, f"secII_acc_{VEHICLE}.csv")
GYR_FILE = os.path.join(PROJECT, f"secII_gyr_{VEHICLE}.csv")


def save_plot(filename: str):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_xyz_csv(path: str) -> pd.DataFrame:
    """
     .cvs format [time, x, y, z].
    """
    df = pd.read_csv(path)
    df.columns = ["time", "x", "y", "z"]
    return df


# --------------------[ SECTION I stuff ]--------------------
def fit_linear_bias(df: pd.DataFrame) -> dict:
    """
    from section 1
    """
    t = df["time"].to_numpy()
    biases = {}
    for axis in ["x", "y", "z"]:
        # np.polyfit returns [slope, intercept] for degree=1
        slope, intercept = np.polyfit(t, df[axis].to_numpy(), 1)
        biases[axis] = (slope, intercept)
    return biases


def correct_gravity_on_z(acc_df: pd.DataFrame) -> pd.DataFrame:
    out = acc_df.copy()
    out["z"] = out["z"] - G
    return out


# --------------------[ SECTION II ]--------------------
def apply_bias_removal(df: pd.DataFrame, biases: dict) -> pd.DataFrame:
    """
    correct bias using b(t) = b_s * t + b0 for each axis.
    """
    t = df["time"].to_numpy()
    out = df.copy()
    for axis in ["x", "y", "z"]:
        b_s, b0 = biases[axis]
        out[axis] = out[axis] - (b_s * t + b0)
    return out


def Cumulative_trapz(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    integration in real time using trapezoid rule inteasd of cumulative sum for lower drift
    
    Cumulative trapezoid integral with same length output and y[0] baseline 0.
    v[k] = v[k-1] + (y[k-1] + y[k]) * 0.5 * (t[k] - t[k-1])
    """
    out = np.zeros_like(y)
    if len(y) > 1:
        dt = np.diff(t)
        out[1:] = np.cumsum((y[:-1] + y[1:]) * 0.5 * dt)
    return out


def kinematics_from_accel_z(acc_df: pd.DataFrame):
    t = acc_df["time"].to_numpy()
    az = acc_df["z"].to_numpy()
    # integration in discret time
    vz = Cumulative_trapz(az, t)
    pz = Cumulative_trapz(vz, t)
    return t, az, vz, pz


def angles_from_gyro_z(gyr_df: pd.DataFrame):
    """
    integrate angular rate to get angle
    """
    t = gyr_df["time"].to_numpy()
    wz = gyr_df["z"].to_numpy()
    theta = Cumulative_trapz(wz, t)
    # theta = np.unwrap(theta)  # possibly if better
    return t, wz, theta


def plot_series(t, y, title, ylab, filename):
    plt.figure()
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel(ylab)
    plt.grid(True)
    save_plot(filename)



def main():
    #--- Section I  ---
    sec1_acc = load_xyz_csv(SEC1_ACC)
    sec1_gyr = load_xyz_csv(SEC1_GYR)
    sec1_acc_gcorr = correct_gravity_on_z(sec1_acc)
    acc_bias = fit_linear_bias(sec1_acc_gcorr)  # {'x':(bs,b0), 'y':(...), 'z':(...)}
    gyr_bias = fit_linear_bias(sec1_gyr)

    # --- Section II  ---
    acc = load_xyz_csv(ACC_FILE)
    gyr = load_xyz_csv(GYR_FILE)

    # --- remove bias and gravity ---
    acc = correct_gravity_on_z(acc)
    acc = apply_bias_removal(acc, acc_bias)
    gyr = apply_bias_removal(gyr, gyr_bias)
    # --- get velo and position  ---
    t_a, az, vz, pz = kinematics_from_accel_z(acc)
    t_g, wz, th = angles_from_gyro_z(gyr)

    # --- plot ---
    prefix = f"vehicle_{VEHICLE}"
    plot_series(t_a, pz,  f"Vehicle {VEHICLE} - position p_z(t)",      "Position z [m]", prefix + "_pz.png"     )
    plot_series(t_a, vz,  f"Vehicle {VEHICLE} - speed v_z(t)",         "Speed z [m/s]",  prefix + "_vz.png"     )
    plot_series(t_a, az,  f"Vehicle {VEHICLE} - acceleration a_z(t)",  "Acceleration z [m/s²]", prefix + "_az.png"    )
    plot_series(t_g, th,  f"Vehicle {VEHICLE} - angle θ_z(t)",         "Angle [rad]",       prefix +    "_theta.png"  )
    plot_series(t_g, wz,  f"Vehicle {VEHICLE} - angular rate ω_z(t)",  "Angular rate [rad/s]", prefix + "_omega.png"  )


if __name__ == "__main__":
    main()
