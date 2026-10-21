# %%
"""
Figure 2 (fig:CIR)
BCMMC estimator of the CIR propagator with Wiener bridges (dots) vs. the exact
formula (solid line). Panel a: linear scale; panel b: logarithmic y-axis.

Outputs: figures/fig2a_CIR_propagator.pdf
         figures/fig2b_CIR_propagator_log.pdf
(The two panels were placed side by side to form figures/BDMM_propagator.pdf in the paper.)
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

# %% Parameters and exact propagator
D = 0.5   # Diffusion coefficient
mu = 0.3  # mean
k = 1.0   # strength of the restoring force (inverse of correlation time)
dt = 1.0
x0 = mu
xmin = max(0.001, mu - 5 * D**2 * sqrt(k))
xmax = mu + 5 * D**2 * sqrt(k)
xs = np.linspace(xmin, xmax, 200)
props_BDMM = f.propagator_BDMM(xs, t=dt, x0=x0, mu=mu, k=k, D=D)

# %% BCMMC estimator with Wiener bridges
model = 1  # CIR // BDMM // DE
xs_BCM = np.linspace(xmin, xmax, 30)
Nbridges = 10
dt_bridge = 0.01
realiz = 100
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

# %% Panel a: linear scale
f.Plot_bonito(xlabel=r'$x$', ylabel=r'$\rho_{t|s}^{\text{CIR}}(x| y)$')
plt.scatter(xs_BCM / mu, prop_BCM_mean, color='red', label='BDMM Propagator (BCM)', zorder=5, s=10, alpha=0.5)
plt.errorbar(xs_BCM / mu, prop_BCM_mean, fmt="none", yerr=2 * prop_BCM_std, color='red', zorder=5, capsize=4, alpha=0.4)
plt.plot(xs / mu, props_BDMM, color="black", zorder=4, lw=1.0)
out = FIG_DIR / "fig2a_CIR_propagator.pdf"
plt.savefig(out, transparent=True, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# %% Panel b: logarithmic y-axis
f.Plot_bonito(xlabel=r'$x$', ylabel=r'$\rho_{t|s}^{\text{CIR}}(x| y)$')
plt.scatter(xs_BCM / mu, prop_BCM_mean, color='red', label='BDMM Propagator (BCM)', zorder=5, s=10, alpha=0.5)
plt.errorbar(xs_BCM / mu, prop_BCM_mean, fmt="none", yerr=2 * prop_BCM_std, color='red', zorder=5, capsize=4, alpha=0.4)
plt.plot(xs / mu, props_BDMM, color="black", zorder=4, lw=1.0)
plt.yscale("log")
plt.yticks([1e-4, 1e-2, 1])
out = FIG_DIR / "fig2b_CIR_propagator_log.pdf"
plt.savefig(out, transparent=True, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# %%
