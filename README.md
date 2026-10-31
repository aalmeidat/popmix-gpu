# popmix-gpu

GPU tools for **population mixture modeling with per-star measurement errors**, built for decomposing nearby galaxies (the Magellanic Clouds, dwarf spheroidals) into stellar populations and for testing whether a dwarf galaxy needs a separate, extended outer component (a stellar halo).

Everything is written in PyTorch and runs on a CPU or a single NVIDIA GPU. This is research code under active development.

## What is in here

| Part | What it does |
| --- | --- |
| `proto/xd.py` | **Extreme deconvolution** (Bovy, Hogg & Roweis 2011): a Gaussian mixture fitted to noisy data with a full error covariance per star, chunked for GPUs. Restarts, BIC over K, bootstrap errors (`scripts/fit_xd.py`). |
| `proto/flow.py` | **Deconvolving normalizing flow** (neural spline flow, [zuko](https://github.com/probabilists/zuko)): maximizes a Monte Carlo estimate of log p(x_i) = log E over e ~ N(0, S_i) of q(x_i − e), so the learned density is the error-free one (`scripts/fit_flow.py`; `--plain` trains an ordinary flow on the noisy data for comparison). |
| `scripts/gap_stat.py`, `scripts/stability.py` | Model-selection checks: a likelihood gap statistic, and bootstrap stability of the per-star assignments (adjusted Rand index). |
| `scripts/benchmark.py` | Times XD and the flow from 10⁴ to 10⁶ stars and extrapolates GPU-hours for a planned workload. |
| `dwarf/` | **Dwarf-halo search.** A mixture of a Milky Way foreground and one or two elliptical Plummer components (main body + bounded extended component), combining sky position, Gaia parallax and proper motions with full covariances, the color–magnitude diagram, and XP log g (with missing values). The foreground template comes from an outer annulus of the same field, with a fitted planar gradient. `fit_dwarf2.py` is the CPU reference; `fit_dwarf_gpu.py` is the same model in PyTorch with exact gradients (see below). |
| `dwarf/pull_gaia_dwarf.py` | Downloads a Gaia DR3 cone (split into declination strips under the archive's row cap) and optionally merges the Zhang, Green & Rix (2023) XP parameters. |
| `slurm/`, `rivanna_start.sh` | Slurm job scripts for a university cluster (written for UVA's Rivanna, using its Apptainer PyTorch container). |

## Install

```
pip install -r requirements.txt
```

PyTorch with CUDA is needed for GPU runs. On clusters, a site PyTorch container plus `pip install --user zuko astropy pyyaml scipy matplotlib` works.

## Quick start (mock data, CPU is fine)

Mixture model + flow on a three-component mock:
```
python scripts/make_mock.py --config config_mock.yaml
python scripts/fit_xd.py --config config_mock.yaml --quick
python scripts/fit_flow.py --config config_mock.yaml --quick
python scripts/plot_compare.py --config config_mock.yaml
```
The XD components in `results_mock/xd/xd_summary.json` should match the injected ones in `scripts/make_mock.py`. The numbers in the mock are illustrative, not measurements.

Dwarf-halo model on a Sculptor-like mock (8° field, halo injected) and a null mock:
```
python dwarf/make_mock_dwarf.py --config dwarf/mock8/scl8mock.yaml --fg_density 500
python dwarf/fit_dwarf_gpu.py  --config dwarf/mock8/scl8mock.yaml --target dwarf/mock8/scl8mock_mock_target.fits
python dwarf/make_mock_dwarf.py --config dwarf/mock8null/scl8null.yaml --fg_density 500 --halo_frac 0
python dwarf/fit_dwarf_gpu.py  --config dwarf/mock8null/scl8null.yaml --target dwarf/mock8null/scl8null_mock_target.fits
```
Each writes `fit_gpu.json` with the 1- and 2-component fits, ΔlnL and ΔBIC.

Real data: `python dwarf/pull_gaia_dwarf.py --config dwarf/sculptor8.yaml --xp` (public Gaia DR3), then `python dwarf/fit_dwarf_gpu.py --config dwarf/sculptor8.yaml --plots`.

## On a GPU cluster

```
bash rivanna_start.sh                    # lists your allocations
bash rivanna_start.sh <allocation>       # XD + flow benchmark  -> results_mock/benchmark/gpu_budget.md
bash rivanna_start.sh <allocation> dwarf # dwarf-halo checks, timings and GPU-hour estimate
```
The scripts find the site's PyTorch container, install the extra packages once, and submit one Slurm job. Edit `slurm/common.sh` and the `#SBATCH` lines for another cluster.

## Notes and known limitations

- **CPU vs GPU dwarf model.** `fit_dwarf_gpu.py --compat` reproduces the CPU model (nearest-cell CMD lookup) and was checked against it on the halo mock (log-likelihood within 2 units). The default mode reads the CMD and log g densities with linear interpolation so that every parameter has a usable gradient.
- **Halo color shift.** The extended component's CMD is the main body's, shifted in BP−RP by a fitted amount (|shift| < 0.15 mag). In the CPU version this shift had a zero finite-difference gradient and never moved from 0; the GPU version fits it. On a mock with an injected −0.08 mag shift it recovers the sign but only about half the amount (−0.035), so treat fitted shifts as lower limits for now.
- **XD** here has no selection-function term (unlike pyGMMis). The **flow** objective is a lower bound that tightens with more noise draws per star; check results are stable when `mc_samples` is doubled.
- `--quick` runs and CPU timings are pipeline tests only.

## Measured performance

On an NVIDIA A100 (40 GB): one XD EM iteration on 10⁶ stars in four dimensions took 0.124 s. GPU-hour estimates for a planned workload are produced by `scripts/benchmark.py` and `dwarf/bench_dwarf_gpu.py`; edit the `budget` blocks to match your plan.

## Citation

If you use this code, please cite it via `CITATION.cff` (GitHub: "Cite this repository") and the methods it implements: Bovy, Hogg & Roweis (2011, AnnAS 5, 1657) for extreme deconvolution, Durkan et al. (2019) for neural spline flows, and Zhang, Green & Rix (2023, MNRAS 524, 1855) if you use the XP parameters.

## License

MIT (see `LICENSE`).
