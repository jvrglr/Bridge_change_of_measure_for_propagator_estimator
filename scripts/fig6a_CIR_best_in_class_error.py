# %%
"""
Figure 6a (fig:SLM_optimal_OU_bridges, panel a)
Error of the BCMMC estimator of the CIR propagator with OU bridges over a grid
of bridge parameters (mu_B, k_B) centred at the best-in-class approximation
(Match_OU_bridge_params), ranked in increasing order.

Output: figures/fig6a_CIR_OU_bridge_error_profile.pdf
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
D = 0.1   # Diffusion coefficient
mu = 1.0  # mean
k = 1.0   # strength of the restoring force (inverse of correlation time)
dt = 1 / k
model = 1  # BDM // CIR
x0 = mu
t0 = 0.0
params = (mu, k, D)

x = mu + 0.0001  # target point where the propagator estimator is evaluated

Nbridges = 1
dt_bridge = 0.01
realiz = 1000  # repetitions used at each (mu_OU, k_OU)

mu_OU_opt, k_OU_opt = f.Match_OU_bridge_params(x, x0, params, model)
print(f"optimal computed mu_OU = {mu_OU_opt:.3f}, optimal k_OU = {k_OU_opt:.3f}")
# insert the theoretical optimum as an actual grid point, so it is evaluated
# with the same MC procedure as everything else
mu_OU_vals = np.sort(np.append(np.linspace(mu_OU_opt - 20, mu_OU_opt + 20, 30), mu_OU_opt))
k_OU_vals = np.sort(np.append(np.linspace(k_OU_opt - 3, k_OU_opt + 3, 30), k_OU_opt))  # k_OU=0 -> Wiener bridge
jm_opt = np.argmin(np.abs(mu_OU_vals - mu_OU_opt))
jk_opt = np.argmin(np.abs(k_OU_vals - k_OU_opt))
jk_wiener = np.argmin(np.abs(k_OU_vals - 0.0))

# %% Scan the (mu_OU, k_OU) grid
error_grid = np.zeros((len(k_OU_vals), len(mu_OU_vals)))

real_prop = f.propagator_BDMM(x, dt, x0, mu, k, D)
for jk, k_OU in enumerate(k_OU_vals):
    for jm, mu_OU in enumerate(mu_OU_vals):
        props = np.zeros(realiz)
        for r in range(realiz):
            if k_OU == 0.0:  # OU bridge degenerates at k_OU=0, use the Wiener bridge instead
                empirical_prop, _, _ = f.prop_BCM_Wiener_bridges(x, dt, x0, t0, np.array([mu, k, D]), model, Nbridges, dt_bridge)
            else:
                empirical_prop, _, _ = f.prop_BCM_OU_bridges(x, dt, x0, t0, np.array([mu, k, D]), model, Nbridges, dt_bridge, params_B=(mu_OU, k_OU))
            props[r] = empirical_prop
        error_grid[jk, jm] = np.abs(np.mean(props) - real_prop)
error_Wiener = error_grid[jk_wiener, :].mean()  # Wiener bridge row doesn't depend on mu_OU, average it down

# %% Error profile: sorted error values across the (mu_OU, k_OU) grid
error_sorted = np.sort(error_grid.flatten())
rank = np.arange(1, len(error_sorted) + 1)
f.Plot_bonito(xlabel='rank', ylabel='Relative error')
plt.scatter(rank, error_sorted, color='black', marker='o', s=5, alpha=0.6)
plt.plot(rank, error_sorted, color='black', lw=0.5, alpha=0.6)
plt.axhline(error_grid[error_grid > 0].min(), color='blue', ls='-', label='grid minimum')
plt.axhline(error_grid[jk_opt, jm_opt], color='red', ls=':', label='optimal', lw=4)
plt.axhline(error_Wiener, color='gray', ls='--', label='Wiener bridge')
plt.yscale('log')
plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1E1])
out = FIG_DIR / "fig6a_CIR_OU_bridge_error_profile.pdf"
plt.savefig(out, transparent=True, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# %%
