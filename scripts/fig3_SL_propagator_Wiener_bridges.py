# %%
"""
Figure 3 (fig:SLM)
BCMMC estimator with Wiener bridges (dots) and crude Monte Carlo estimator
(bars) of the stochastic logistic (SL) propagator, logarithmic y-axis.

Output: figures/fig3_SL_propagator_log.pdf
"""
from pathlib import Path

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

import Functions as f

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 1
f.set_seed(SEED)

# %% Crude Monte Carlo reference
D = 0.1   # Diffusion coefficient
mu = 1.0  # mean
k = 1.0   # strength of the restoring force (inverse of correlation time)
dt = 1.0
model = 2  # SLM
x0 = mu
t0 = 0.0
dt_int = 0.01
realiz = 10000
npoints = 50
params = (mu, k, D)

bins_crude, prop_crude_MC = f.Estimate_propagator_crude_MC(x0, t0, dt, dt_int, realiz, npoints, params, model)
dx_crude = bins_crude[1] - bins_crude[0]

# %% BCMMC estimator with Wiener bridges
model = 2  # SLM // EN
xmin = 0.7
xmax = 1.4
xs_BCM = np.linspace(xmin, xmax, 30)
Nbridges = 10
dt_bridge = 0.01
realiz = 1

prop_BCM_mean = np.zeros(len(xs_BCM))
prop_BCM_std = np.zeros(len(xs_BCM))

for r in range(realiz):
    prop_BCM = np.zeros_like(xs_BCM)
    for i, x in enumerate(xs_BCM):
        empirical_prop, _, _ = f.prop_BCM_Wiener_bridges(x, dt, x0, 0.0, np.array([mu, k, D]), model, Nbridges, dt_bridge)
        prop_BCM[i] = empirical_prop
    prop_BCM_mean += prop_BCM
    prop_BCM_std += prop_BCM**2
prop_BCM_mean /= realiz
prop_BCM_std /= realiz
prop_BCM_std = np.sqrt(prop_BCM_std - prop_BCM_mean**2) / sqrt(realiz)

# %% Plot
f.Plot_bonito(xlabel=r'$x$', ylabel=r'$\rho_{t|s}^{\text{SLM}}(x| y)$')
plt.scatter(xs_BCM / mu, prop_BCM_mean, color='red', label='SLM Propagator (BCM)', zorder=5, s=10, alpha=0.5)
plt.bar(bins_crude / mu, prop_crude_MC, width=dx_crude, color="dodgerblue", zorder=4, lw=1.0)
plt.yscale("log")
plt.yticks([1e-5, 1e-3, 1e-1, 10])
out = FIG_DIR / "fig3_SL_propagator_log.pdf"
plt.savefig(out, transparent=True, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# %%
