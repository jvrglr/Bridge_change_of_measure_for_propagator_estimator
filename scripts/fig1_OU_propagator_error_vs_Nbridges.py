# %%
"""
Figure 1 (fig:OU_propagator)
Relative error of the BCMMC estimator of the OU propagator with Wiener bridges,
as a function of the number of bridges n_B, for three bridge discretizations.

By default the figure is drawn from the precomputed data in ../data/.
Set RECOMPUTE = True to regenerate those files first (takes several minutes).

Output: figures/fig1_OU_propagator_error_vs_Nbridges.pdf
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import Functions as f

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

RECOMPUTE = False  # True -> regenerate data/OU_propagator_error_dt*.dat
SEED = 1

# %% Additive OU process: parameters
D = 1.0   # Diffusion coefficient
mu = 1.0  # mean
k = 1.0   # strength of the restoring force (inverse of correlation time)
tf = 1.0
t0 = 0.0
x0 = mu
xf = mu + 0.1
model = 0  # OU

tfmt0 = tf - t0
target_prop = f.propagator_OU_process(xf, tfmt0, x0, mu, k, D)

# %% Convergence study: average relative error vs Nbridges
Nbridges_list = np.logspace(1, 5, 10).astype(int)
dt_bridges = np.logspace(-3, -1, 3)
Nrealizations = 100

if RECOMPUTE:
    f.set_seed(SEED)
    for dt_bridge in dt_bridges:
        print(f"dt_bridge = {dt_bridge}")
        mean_errors = []
        std_errors = []
        for Nbridges in Nbridges_list:
            errors = []
            for _ in range(Nrealizations):
                empirical_prop, _, _ = f.prop_BCM_Wiener_bridges(xf, tfmt0, x0, 0.0, np.array([mu, k, D]), model, Nbridges, dt_bridge)
                rel_error = abs(empirical_prop - target_prop) / target_prop * 100
                errors.append(rel_error)
            mean_errors.append(np.mean(errors))
            std_errors.append(np.std(errors))
        np.savetxt(DATA_DIR / f"OU_propagator_error_dt{dt_bridge:.3f}.dat",
                   np.column_stack((Nbridges_list, mean_errors, std_errors)))

# %% Plotting the results: relative error
colors = ['dodgerblue', 'orange', 'green']
markers = ['s', 'o', 'D']
f.Plot_bonito(xlabel=r"$\overline{n}$", ylabel=r"Relative error", x_size=4, y_size=3)
for jj, dt_bridge in enumerate(dt_bridges):
    filename = DATA_DIR / f"OU_propagator_error_dt{dt_bridge:.3f}.dat"
    Nbridges_list, mean_errors, std_errors = np.loadtxt(filename, unpack=True)
    mean_errors = mean_errors / 100      # from % to number <1
    std_errors = std_errors / 100 * 2    # from % to number <1 and 95% interval
    exponent = int(round(np.log10(dt_bridge)))  # e.g. 0.1 -> -1, 0.01 -> -2
    label = f"$\\Delta t_B = 10^{{{exponent}}}$"
    yerr_lo = mean_errors * (1 - np.exp(-std_errors / mean_errors))
    yerr_hi = mean_errors * (np.exp(std_errors / mean_errors) - 1)
    plt.errorbar(Nbridges_list, mean_errors, yerr=[yerr_lo, yerr_hi], fmt='none', color=colors[jj],
                 capsize=3, capthick=1.0, elinewidth=1.5, alpha=0.5)
    plt.scatter(Nbridges_list, mean_errors, color=colors[jj], s=20, label=label, marker=markers[jj], zorder=3)
    plt.plot(Nbridges_list, mean_errors, color=colors[jj], alpha=0.2)

# 1/sqrt(N) reference line
N_arr = np.array(Nbridges_list, dtype=float)
ref = mean_errors[0] * np.sqrt(Nbridges_list[0]) / np.sqrt(N_arr)
plt.plot(N_arr, 0.04 * ref, ls='--', linewidth=1.5, color="black")

plt.xscale('log')
plt.yscale('log')
plt.xticks([10, 100, 1000, 10000, 1e5])
plt.yticks([1e-4, 0.001, 0.01, 0.1])
plt.ylim(0.5e-5, 0.3)
plt.legend(fontsize=10, frameon=False)
out = FIG_DIR / "fig1_OU_propagator_error_vs_Nbridges.pdf"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# %%
