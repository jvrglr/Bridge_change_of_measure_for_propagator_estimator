# %%
"""
Figure 7 (fig:discretization)
Relative error of the BCMMC estimator of the CIR propagator with OU bridges
(best-in-class parameters) vs. the time window t, comparing the left-point
discretization of the Girsanov integral (black circles) with the trapezoidal
one (red squares).

Output: figures/fig7_changing_discretization.pdf
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import Functions as f

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 1
f.set_seed(SEED)

# %% Parameters
D = 0.1   # Diffusion coefficient
mu = 1.0  # mean
k = 1.0   # strength of the restoring force (inverse of correlation time)
model = 1  # BDM // CIR
x0 = mu
t0 = 0.0
params = (mu, k, D)
x = mu + 0.1  # target point where the propagator is evaluated

# %% Relative error vs time window
dts = np.logspace(np.log10(0.001), np.log10(100), 20)
errors = np.zeros(len(dts)) - 1
errors_trapezoid = np.zeros(len(dts)) - 1
N_bridges = 10000

for i, dt in enumerate(dts):
    true_prop = f.propagator_BDMM(x, dt, x0, mu, k, D)
    dt_bridge = dt / 100
    prop_BCMMC, _, _ = f.prop_BCM_OU_bridges_optimal_params(x, dt, x0, t0, params, model, N_bridges, dt_bridge, errors=False)
    prop_BCMMC_trapezoid, _, _ = f.prop_BCM_OU_bridges_optimal_params_trapezoid(x, dt, x0, t0, params, model, N_bridges, dt_bridge, errors=False)
    errors[i] = abs(true_prop - prop_BCMMC) / true_prop
    errors_trapezoid[i] = abs(true_prop - prop_BCMMC_trapezoid) / true_prop

# %% Plot
f.Plot_bonito(xlabel=r'$t$', ylabel='relative error')
plt.scatter(dts, errors, color='black', s=10)
plt.plot(dts, errors, color='black', lw=1, alpha=0.1)
plt.scatter(dts, errors_trapezoid, color='red', s=10, marker="s")
plt.plot(dts, errors_trapezoid, color='red', lw=1, alpha=0.1)
plt.xscale('log')
plt.yscale('log')
plt.yticks(np.logspace(-10, 0, 6))
plt.xticks(np.logspace(-3, 2, 6))
out = FIG_DIR / "fig7_changing_discretization.pdf"
plt.savefig(out, bbox_inches='tight', transparent=True)
plt.close()
print(f"Saved {out}")

# %%
