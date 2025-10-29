import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import inv
import os

# constant for g and data files path
G = 9.805  # m/s^2
ACC_FILE = "projectfiles/secI_acc.csv"
GYR_FILE = "projectfiles/secI_gyr.csv"

# to save plots
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def save_plot(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# read data from csv and create dataframe for them
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['time', 'x', 'y', 'z']
    return df

# @TODO : add this model to the doc in the math section
def fit_bias(df):
    # linear model: bias = b0 + b_s*t
    t = df['time'].values
    biases = {}
    for axis in ['x', 'y', 'z']:
        coeffs = np.polyfit(t, df[axis], 1)
        biases[axis] = coeffs  
    return biases

def plot_bias(df, biases, title):
    t = df['time']
    for axis in ['x', 'y', 'z']:
        plt.figure()
        plt.plot(t, df[axis], label=f'{axis}-axis data')
        plt.plot(t, np.polyval(biases[axis], t), 'r--', label='Fitted bias line')
        plt.title(f'{title} {axis}-axis')
        plt.xlabel('Time [s]')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        save_plot(f"{title.lower()}_{axis}_bias.png")

# correct accelerometer data for gravity constant in z axis
def correct_gravity(df):
    df_corr = df.copy()
    df_corr['z'] = df_corr['z'] - G
    return df_corr

def compute_stats(df):
    mean = df[['x', 'y', 'z']].mean()
    var = df[['x', 'y', 'z']].var()
    cov = df[['x', 'y', 'z']].cov()
    return mean, var, cov

def plot_histograms(df, mean, var, sensor):
    for axis in ['x', 'y', 'z']:
        data = df[axis]
        mu, sigma = mean[axis], np.sqrt(var[axis])
        plt.figure()
        plt.hist(data, bins=50, density=True, alpha=0.6, color='gray')
        x_vals = np.linspace(data.min(), data.max(), 100)
        plt.plot(x_vals, norm.pdf(x_vals, mu, sigma), 'r--')
        plt.title(f'{sensor} {axis}-axis noise histogram')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid()
        save_plot(f"{sensor.lower()}_{axis}_hist.png")

def main():
    acc = load_data(ACC_FILE)
    gyr = load_data(GYR_FILE)
    
    acc_corr = correct_gravity(acc)
    
    acc_bias = fit_bias(acc_corr)
    gyr_bias = fit_bias(gyr)

    plot_bias(acc_corr, acc_bias, 'Accelerometer')
    plot_bias(gyr, gyr_bias, 'Gyroscope')

    acc_mean, acc_var, acc_cov = compute_stats(acc_corr)
    gyr_mean, gyr_var, gyr_cov = compute_stats(gyr)
    
    print("Accelerometer mean:\n", acc_mean)
    print("Accelerometer variance:\n", acc_var)
    print("Accelerometer covariance:\n", acc_cov, "\n")

    print("Gyroscope mean:\n", gyr_mean)
    print("Gyroscope variance:\n", gyr_var)
    print("Gyroscope covariance:\n", gyr_cov, "\n")

    plot_histograms(acc_corr, acc_mean, acc_var, 'Accelerometer')
    plot_histograms(gyr, gyr_mean, gyr_var, 'Gyroscope')

if __name__ == "__main__":
    main()
