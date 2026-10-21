# %%
"""
Figure 4 (fig:OU_OU_bridges)
Standard deviation of the BCMMC estimator of the OU propagator with OU bridges,
as a function of the ratio between bridge and target means, mu_B / mu.
Target and bridge OU processes differ only through their means.

The standard deviation is computed over `realiz` independent repetitions of
the estimator, each using `Nbridges` bridges.

Output: figures/fig4_OU_BCM_variance_vs_mu_OU.pdf
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
D = 1.0    # Diffusion coefficient (target); the OU bridge has unit diffusion
mu = 1.0   # mean (target)
k = 1.0    # strength of the restoring force (target)
k_OU = 1.0 # strength of the restoring force (bridge), equal to the target one
model = 0  # OU
y = mu     # initial point
s = 0.0
t = 1.0
x = 0.1    # propagator argument

Nbridges = 1000
dt_bridge = 0.01
realiz = 100  # independent repetitions of the estimator used to compute its standard deviation

mu_OU_vals = mu * np.linspace(1.0, 8.0, 10)

# %% Standard deviation of the estimator vs mu_OU
sigmas = np.zeros(len(mu_OU_vals))
for j, mu_OU in enumerate(mu_OU_vals):
    props = np.zeros(realiz)
    for r in range(realiz):
        empirical_prop, _, _ = f.prop_BCM_OU_bridges(x, t, y, s, np.array([mu, k, D]), model, Nbridges, dt_bridge, params_B=(mu_OU, k_OU))
        props[r] = empirical_prop
    sigmas[j] = np.std(props)
    print(f"mu_OU/mu = {mu_OU / mu:.2f}  sigma = {sigmas[j]:.4g}")

# %% Plot
f.Plot_bonito(xlabel=r'$\overline{\mu}\ /\mu$', ylabel=r'$\sigma(\hat{\rho}^{\mathrm{BCM}})$')
plt.scatter(mu_OU_vals / mu, sigmas, color='black', s=10, zorder=3)
out = FIG_DIR / "fig4_OU_BCM_variance_vs_mu_OU.pdf"
plt.savefig(out, transparent=True, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# %%
