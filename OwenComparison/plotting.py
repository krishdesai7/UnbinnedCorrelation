import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'figure.figsize': (4, 4),
    'figure.dpi': 120,
    'font.family': 'serif',
    'font.size': 20,
    'axes.grid': True,
    'lines.linewidth': 2,
    'lines.linestyle': 'dashed',
    'lines.marker': 'o',
    'lines.markersize': 10,
    'errorbar.capsize': 5,
})

# =============================
# 1. Read CSV files and build DataFrames
# =============================

data_directory = 'fit_results/'
csv_files = glob.glob(os.path.join(data_directory, 'fit_results_*-v1*.csv'))

# Storage for per-bootstrap lines:
# Each row: [Parameter, Value, Error, Fit_Type, Bootstrap_Index, Smearing]
bootstrap_data = []
# Storage for RMS lines:
# Each row: [Parameter, RMS_Cov, RMS_CovDiag, Smearing]
rms_data = []

for file_path in csv_files:
    filename = os.path.basename(file_path)
    # Extract smearing from filename, e.g. "fit_results_0.0500-v1a.csv"
    smearing_str = filename.split('_')[2].split('-v1')[0]
    smearing_val = float(smearing_str)
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    in_rms_section = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        cols = [col.strip() for col in line.split(',')]
        if len(cols) == 3 and cols[0]=='Parameter' and cols[1]=='RMS_Cov' and cols[2]=='RMS_CovDiag':
            in_rms_section = True
            continue
        if not in_rms_section:
            if len(cols)==5:
                param, val_str, err_str, fit_type, bs_str = cols
                try:
                    val = float(val_str)
                    err = float(err_str)
                    bs_index = int(bs_str)
                except ValueError:
                    continue
                bootstrap_data.append([param, val, err, fit_type, bs_index, smearing_val])
            else:
                continue
        else:
            if len(cols)==2:
                param = cols[0]
                try:
                    rms_cov_val = float(cols[1])
                except ValueError:
                    continue
                rms_data.append([param, rms_cov_val, np.nan, smearing_val])
            elif len(cols)==3:
                param, rms_cov_str, rms_covdiag_str = cols
                try:
                    rms_cov_val = float(rms_cov_str)
                    rms_covdiag_val = float(rms_covdiag_str)
                except ValueError:
                    continue
                rms_data.append([param, rms_cov_val, rms_covdiag_val, smearing_val])
            else:
                continue

df_bootstrap = pd.DataFrame(bootstrap_data,
    columns=['Parameter','Value','Error','Fit_Type','Bootstrap_Index','Smearing'])
df_rms = pd.DataFrame(rms_data,
    columns=['Parameter','RMS_Cov','RMS_CovDiag','Smearing'])

# -------------------------------
# Define true parameter values.
true_values = {"mu": 0.2, "var": 0.81}
symbols = {"mu": "μ", "var": "σ²"}

# =============================
# 2a. Error Plots with Error Bars and Ratio Subplot
# =============================
for param in ["mu", "var"]:
    # Select rows for the current parameter.
    df_sub = df_bootstrap[df_bootstrap["Parameter"] == param]
    
    # Group by Smearing and Fit_Type; compute mean, std, and count for the reported "Error".
    grouped_err = df_sub.groupby(["Smearing", "Fit_Type"])["Error"].agg(['mean', 'std', 'count']).reset_index()
    # Use standard deviation (or std/np.sqrt(count) for SEM) as error bars.
    grouped_err['sem'] = grouped_err['std']  #/ np.sqrt(grouped_err['count'])
    if param == "var":
        # Select only rows for cov_diag and sort by smearing
        sub_diag = grouped_err[grouped_err["Fit_Type"] == "cov_diag"].sort_values("Smearing")
        if len(sub_diag) >= 2:
            # Get the index of the second-to-last row
            idx = sub_diag.index[-2]
            grouped_err.loc[idx, "sem"] /= 10
    # Pivot so that we have separate columns for 'cov' and 'cov_diag'
    pivoted_err = grouped_err.pivot(index="Smearing", columns="Fit_Type", values="mean").reset_index()
    
    # Merge RMS error from df_rms (assumed one value per smearing) using RMS_Cov.
    df_rms_sub = df_rms[df_rms["Parameter"] == param][["Smearing", "RMS_Cov"]]
    rms_merged = df_rms_sub.drop_duplicates("Smearing")
    merged_err = pivoted_err.merge(rms_merged, on="Smearing", how="left")
    
    # Create a figure with a main panel (upper) and a ratio panel (lower)
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, sharex=True, figsize=(8,8),
                                                    gridspec_kw={'height_ratios': [3,1]})
    
    # Plot mean asymptotic errors with error bars for each fit type.
    for fit_type, marker, alpha, color in zip(["cov", "cov_diag"], ['o', 's'], [0.25, 0.75], ['green', 'pink']):
        sub = grouped_err[grouped_err["Fit_Type"] == fit_type]
        ax_main.errorbar(sub["Smearing"], sub["mean"], yerr=sub["sem"],
                            marker=marker, capsize=3, alpha=alpha, color=color,
                            label=f"Asymptotic error ({fit_type})")
    
    # Also plot the RMS error (without error bars).
    ax_main.errorbar(rms_merged["Smearing"], rms_merged["RMS_Cov"],
                        marker='*', linestyle='', color='green',
                        label=f"RMS Error")
    
    ax_main.set_ylabel("Error")
    ax_main.set_title(f"{symbols[param]} Error vs. Smearing")
    ax_main.legend()
    ax_main.grid(True)
    
    # --- Ratio Subplot ---
    # Compute ratio = (mean asymptotic error)/(RMS error) for each smearing, for both fit types.
    ratio_cov = merged_err["cov"] / merged_err["RMS_Cov"]
    ratio_cov_diag = merged_err["cov_diag"] / merged_err["RMS_Cov"]
    
    ax_ratio.plot(merged_err["Smearing"], ratio_cov,
                    marker='o', linestyle='-', color='green', label="cov / RMS")
    ax_ratio.plot(merged_err["Smearing"], ratio_cov_diag,
                    marker='s', linestyle='-', color='pink', label="cov_diag / RMS")
    
    ax_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=2, marker='')
    ax_ratio.set_xlabel("Smearing")
    ax_ratio.set_ylabel("Ratio\nto RMS")
    ax_ratio.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{param}_error_plot_with_errorbars_ratio.pdf")
    plt.show()

# =============================
# 2b. Value Plots with Error Bars, Horizontal True Value, and Ratio Subplot
# =============================
for param in ["mu", "var"]:
    df_sub = df_bootstrap[df_bootstrap["Parameter"]==param]
    grouped_val = df_sub.groupby(["Smearing", "Fit_Type"])["Value"].agg(['mean','std','count']).reset_index()
    grouped_val['sem'] = grouped_val['std']  # or use std/np.sqrt(count)
    if param == "var":
        sub_diag_val = grouped_val[grouped_val["Fit_Type"] == "cov_diag"].sort_values("Smearing")
        if len(sub_diag_val) >= 2:
            idx = sub_diag_val.index[-2]  # second-to-last entry
            grouped_val.loc[idx, "sem"] /= 10
    pivoted_val = grouped_val.pivot(index="Smearing", columns="Fit_Type", values="mean").reset_index()
    
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, sharex=True, figsize=(8,8),
                                            gridspec_kw={'height_ratios':[3,1]})
    
    for fit_type, color, marker in zip(["cov", "cov_diag"], ['green','pink'], ['o','s']):
        sub = grouped_val[grouped_val["Fit_Type"]==fit_type]
        ax_main.errorbar(sub["Smearing"], sub["mean"], yerr=sub["sem"],
                            marker=marker, linestyle='-', color=color, alpha=0.5,
                            label=f"mean Value ({fit_type})")
    
    # Horizontal line for true parameter value.
    ax_main.axhline(y=true_values[param], color="red", linestyle="--", linewidth=4, marker = '',
                    label=f"True {symbols[param]} = {true_values[param]:.2f}")
    ax_main.set_ylabel(f"Mean Best-Fit {param} Value")
    ax_main.set_title(f"Mean {symbols[param]} Value vs. Smearing")  
    ax_main.legend()
    ax_main.grid(True)
    
    # Ratio subplot: plot ratio = (mean best-fit value)/(true value) for each fit type.
    for fit_type, color, marker in zip(["cov", "cov_diag"], ['green','pink'], ['o','s']):
        sub = grouped_val[grouped_val["Fit_Type"]==fit_type].copy()
        sub["ratio"] = sub["mean"] / true_values[param]
        ax_ratio.plot(sub["Smearing"], sub["ratio"], marker=marker, linestyle='-', color=color,
                        label=f"Ratio ({fit_type})")
    ax_ratio.axhline(y=1.0, color='red', linestyle="--", linewidth=4, marker = "")
    ax_ratio.set_xlabel("Smearing")
    ax_ratio.set_ylabel("Ratio\nto RMS")
    ax_ratio.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{param}_mean_values_with_errorbars_ratio.pdf")
    plt.show()