# %%
"""
Figure 6b (fig:SLM_optimal_OU_bridges, panel b)
Variance of the BCMMC estimator of the stochastic logistic (SL) propagator with
OU bridges over a grid of bridge parameters (mu_B, k_B), augmented with the
Wiener bridge (k_B = 0) and the best-in-class approximation, ranked in
increasing order.

Output: figures/fig6b_SL_OU_bridge_variance_profile.pdf
(Panels a and b were placed side by side to form figures/Optimal_OU_SLM_and_BDM.pdf in the paper.)
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
D = 0.3   # Diffusion coefficient
mu = 1.0  # mean
k = 3.0   # strength of the restoring force (inverse of correlation time)
dt = 0.75
model = 2  # SLM
x0 = mu
t0 = 0.0
params = (mu, k, D)

x = 0.95  # target point where the propagator estimator variance is evaluated

Nbridges = 1
dt_bridge = 0.01
realiz = 1000  # repetitions used to estimate the variance at each (mu_OU, k_OU)

mu_OU_opt, k_OU_opt = f.Match_OU_bridge_params(x, x0, params, model)
print(f"optimal mu_OU = {mu_OU_opt:.3f}, optimal k_OU = {k_OU_opt:.3f}")

# insert the theoretical optimum as an actual grid point, so it is evaluated
# with the same MC procedure as everything else
mu_OU_vals = np.sort(np.append(np.linspace(-0.3, 0.3, 30), mu_OU_opt))
k_OU_vals = np.sort(np.append(np.append(np.linspace(0.5, 8.0, 30), k_OU_opt), 0.0))  # k_OU=0 -> Wiener bridge
jm_opt = np.argmin(np.abs(mu_OU_vals - mu_OU_opt))
jk_opt = np.argmin(np.abs(k_OU_vals - k_OU_opt))
jk_wiener = np.argmin(np.abs(k_OU_vals - 0.0))

# %% Scan the (mu_OU, k_OU) grid
variance_grid = np.zeros((len(k_OU_vals), len(mu_OU_vals)))

for jk, k_OU in enumerate(k_OU_vals):
    for jm, mu_OU in enumerate(mu_OU_vals):
        props = np.zeros(realiz)
        for r in range(realiz):
            if k_OU == 0.0:  # OU bridge degenerates at k_OU=0, use the Wiener bridge instead
                empirical_prop, _, _ = f.prop_BCM_Wiener_bridges(x, dt, x0, t0, np.array([mu, k, D]), model, Nbridges, dt_bridge)
            else:
                empirical_prop, _, _ = f.prop_BCM_OU_bridges(x, dt, x0, t0, np.array([mu, k, D]), model, Nbridges, dt_bridge, params_B=(mu_OU, k_OU))
            props[r] = empirical_prop
        variance_grid[jk, jm] = np.var(props)

variance_opt = variance_grid[jk_opt, jm_opt]  # MC estimate of the variance exactly at the optimum
variance_wiener = variance_grid[jk_wiener, :].mean()  # Wiener bridge row doesn't depend on mu_OU, average it down

# %% Variance profile: sorted variance values across the (mu_OU, k_OU) grid
variance_sorted = np.sort(variance_grid.flatten())
rank = np.arange(1, len(variance_sorted) + 1)

f.Plot_bonito(xlabel='rank', ylabel='Var')
plt.scatter(rank, variance_sorted, color='black', marker='o', s=3, alpha=0.6)
plt.axhline(variance_grid.min(), color='blue', ls='-', label='grid minimum')
plt.axhline(variance_opt, color='red', ls=':', label='optimal', lw=4)
plt.axhline(variance_wiener, color='gray', ls='--', label='Wiener bridge')
plt.yscale('log')
plt.yticks([1e-2, 1e0, 1e2, 1e4])
out = FIG_DIR / "fig6b_SL_OU_bridge_variance_profile.pdf"
plt.savefig(out, transparent=True, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# %%
