# %%
"""
Figure 5 (fig:SLM_OU_bridges)
BCMMC estimator of the stochastic logistic (SL) propagator with OU bridges.
Panel a: MSE between BCMMC and crude MC (reference) estimators vs. the bridge
         parameter k_B.
Panel b: BCMMC estimator for the best- and worst-performing k_B, compared with
         the crude MC reference.

Outputs: figures/fig5a_SL_MSE_vs_kOU.pdf
         figures/fig5b_SL_propagator_log_using_OU_bridges.pdf
(The two panels were placed side by side to form figures/SLM_OU_bridges.pdf in the paper.)
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

# %% Crude Monte Carlo reference
D = 0.3   # Diffusion coefficient
mu = 1.0  # mean
k = 3.0   # strength of the restoring force (inverse of correlation time)
dt = 0.75
model = 2  # SLM
x0 = mu
t0 = 0.0
dt_int = 0.01
realiz = 100000
npoints = 50
params = (mu, k, D)

bins_crude, prop_crude_MC = f.Estimate_propagator_crude_MC(x0, t0, dt, dt_int, realiz, npoints, params, model)

# %% BCMMC with Wiener (k_OU = 0) and OU bridges, scanning k_OU
model = 2  # SLM // EN
xmin = 0.7
xmax = 1.4
xs_BCM = np.linspace(xmin, xmax, 30)
prop_crude_MC_interp = np.interp(xs_BCM, bins_crude, prop_crude_MC)  # crude MC curve resampled onto xs_BCM, for MSE

Nbridges = 1
dt_bridge = 0.01
realiz_MSE = 30  # repetitions averaged per k_OU, to smooth out the MSE(k) curve

mu_OU = -0.05  # mean for the OU bridge (fixed, matches linearization at x0 = mu)
ks = np.linspace(0.0, 15.0, 31)  # k_OU values to scan; k_OU = 0 falls back to the Wiener bridge

MSEs = np.zeros(len(ks))
all_prop_BCM = np.zeros((len(ks), len(xs_BCM)))

for j, k_OU in enumerate(ks):
    prop_BCM_mean = np.zeros(len(xs_BCM))
    for r in range(realiz_MSE):
        prop_BCM = np.zeros_like(xs_BCM)
        for i, x in enumerate(xs_BCM):
            if k_OU == 0.0:
                empirical_prop, _, _ = f.prop_BCM_Wiener_bridges(x, dt, x0, 0.0, np.array([mu, k, D]), model, Nbridges, dt_bridge)
            else:
                empirical_prop, _, _ = f.prop_BCM_OU_bridges(x, dt, x0, 0.0, np.array([mu, k, D]), model, Nbridges, dt_bridge, params_B=(mu_OU, k_OU))
            prop_BCM[i] = empirical_prop
        prop_BCM_mean += prop_BCM
    prop_BCM_mean /= realiz_MSE
    all_prop_BCM[j] = prop_BCM_mean
    MSEs[j] = np.mean((prop_BCM_mean - prop_crude_MC_interp)**2)

idx_best = np.argmin(MSEs)
idx_worst = np.argmax(MSEs)
k_opt = ks[idx_best]
k_worst = ks[idx_worst]
print(f"k_OU that minimizes the MSE: {k_opt}")
print(f"k_OU that maximizes the MSE: {k_worst}")

# Keep only the best- and worst-matching k_OU propagator curves for comparison
prop_BCM_mean_best = all_prop_BCM[idx_best]
prop_BCM_mean_worst = all_prop_BCM[idx_worst]

# %% Panel a: MSE vs k_OU
f.Plot_bonito(xlabel=r'$\overline{k}$', ylabel='MSE')
plt.scatter(ks, MSEs, color='black', label='MSE', zorder=5, s=10, alpha=0.5, marker="o")
plt.plot(ks, MSEs, color='black', ls="--", alpha=0.2)
plt.axvline(k_opt, color='gray', ls=':', alpha=0.6)
plt.yscale("log")
plt.yticks([1e-4, 1e-2, 1])
out = FIG_DIR / "fig5a_SL_MSE_vs_kOU.pdf"
plt.savefig(out, transparent=True, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# %% Panel b: best and worst k_OU vs crude MC
f.Plot_bonito(xlabel=r'$x$', ylabel=r'$\rho_{t|s}^{\text{SLM}}(x| y)$')
plt.scatter(xs_BCM / mu, prop_BCM_mean_best, color='green', label=f'OU bridge, k_OU={k_opt:.1f} (best)', zorder=5, s=10, alpha=0.5, marker="o")
plt.plot(xs_BCM / mu, prop_BCM_mean_best, color='green', ls="--", alpha=0.1)
plt.scatter(xs_BCM / mu, prop_BCM_mean_worst, color='red', label=f'OU bridge, k_OU={k_worst:.1f} (worst)', zorder=5, s=10, alpha=0.5, marker="s")
plt.plot(xs_BCM / mu, prop_BCM_mean_worst, color='red', ls="--", alpha=0.1)
plt.plot(bins_crude / mu, prop_crude_MC, color="black", alpha=0.5)
plt.yscale("log")
plt.xlim([xmin, xmax])
plt.yticks([1e-3, 1e-1, 10])
out = FIG_DIR / "fig5b_SL_propagator_log_using_OU_bridges.pdf"
plt.savefig(out, transparent=True, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# %%
