# Bridge change-of-measure for propagator estimation of diffusions

Code to reproduce the figures of

> J. Aguilar and R. Gatto, *Bridge change-of-measure for propagator estimation of diffusions* (2026).

The paper studies a Monte Carlo estimator of the propagator (transition density) of one-dimensional diffusions based on a bridge change of measure (BCMMC): paths are drawn from an auxiliary Wiener or Ornstein–Uhlenbeck bridge between the two fixed endpoints, and the target propagator is recovered through the Girsanov likelihood ratio after a Lamperti transform.

## Repository layout

```
scripts/
  Functions.py                               # all shared functions (propagators, bridges, estimators, plot style)
  fig1_OU_propagator_error_vs_Nbridges.py
  fig2_CIR_propagator.py
  fig3_SL_propagator_Wiener_bridges.py
  fig4_OU_variance_vs_mu_bridge.py
  fig5_SL_propagator_OU_bridges.py
  fig6a_CIR_best_in_class_error.py
  fig6b_SL_best_in_class_variance.py
  fig7_changing_discretization.py
data/
  OU_propagator_error_dt{0.001,0.010,0.100}.dat   # precomputed data for Fig. 1
figures/                                     # output folder (created by the scripts)
```

## Figures

| Paper figure | Script | Output(s) in `figures/` | Model / bridge |
|---|---|---|---|
| Fig. 1 | `fig1_OU_propagator_error_vs_Nbridges.py` | `fig1_OU_propagator_error_vs_Nbridges.pdf` | OU target, Wiener bridges |
| Fig. 2a, 2b | `fig2_CIR_propagator.py` | `fig2a_CIR_propagator.pdf`, `fig2b_CIR_propagator_log.pdf` | CIR target, Wiener bridges |
| Fig. 3 | `fig3_SL_propagator_Wiener_bridges.py` | `fig3_SL_propagator_log.pdf` | Stochastic logistic target, Wiener bridges vs crude MC |
| Fig. 4 | `fig4_OU_variance_vs_mu_bridge.py` | `fig4_OU_BCM_variance_vs_mu_OU.pdf` | OU target, OU bridges |
| Fig. 5a, 5b | `fig5_SL_propagator_OU_bridges.py` | `fig5a_SL_MSE_vs_kOU.pdf`, `fig5b_SL_propagator_log_using_OU_bridges.pdf` | Stochastic logistic target, OU bridges |
| Fig. 6a | `fig6a_CIR_best_in_class_error.py` | `fig6a_CIR_OU_bridge_error_profile.pdf` | CIR target, OU bridges |
| Fig. 6b | `fig6b_SL_best_in_class_variance.py` | `fig6b_SL_OU_bridge_variance_profile.pdf` | Stochastic logistic target, OU bridges |
| Fig. 7 | `fig7_changing_discretization.py` | `fig7_changing_discretization.pdf` | CIR target, OU bridges (left-point vs trapezoid) |

Multi-panel figures in the paper were assembled from the individual panels produced here.

## Requirements

Python ≥ 3.10 with NumPy, SciPy, Matplotlib and Numba:

```bash
pip install -r requirements.txt
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11 and Numba 0.68.

## Usage

Run any script from the repository root (or from anywhere; paths are resolved relative to the script):

```bash
python scripts/fig2_CIR_propagator.py
```

or generate all figures at once:

```bash
for s in scripts/fig*.py; do python "$s"; done
```

Each script takes roughly one to two minutes on a laptop. The scripts are also split into `# %%` cells, so they can be run interactively in VS Code, Spyder or Jupyter.

**Fig. 1** is drawn from the precomputed files in `data/` by default. To regenerate them, set `RECOMPUTE = True` at the top of `fig1_OU_propagator_error_vs_Nbridges.py` (this takes several minutes).

**Randomness.** Every script fixes a seed (`SEED = 1`) through `Functions.set_seed`, which seeds both NumPy and Numba's internal generator. Re-running a script therefore gives identical output; the figures in the paper were produced without a fixed seed, so they match the regenerated ones up to Monte Carlo fluctuations.

## Citation

If you use this code, please cite:

```bibtex
@article{AguilarGatto2026BridgeCOM,
  author  = {Aguilar, Javier and Gatto, Riccardo},
  title   = {Bridge change-of-measure for propagator estimation of diffusions},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
