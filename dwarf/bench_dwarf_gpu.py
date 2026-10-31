"""Time the dwarf-halo likelihood on the GPU and turn it into GPU-hours for the proposal.

1. Builds the 2-component model on a mock dwarf field (real Gaia-like size, ~150k stars).
2. Times one likelihood + gradient evaluation at increasing star counts (stars resampled from the mock;
   the cost depends on the number of stars, not on which stars they are).
3. Takes the number of evaluations a full fit needs (1- and 2-component models, all EM steps) from the
   fit_*.json files of real GPU fits run earlier in the same job (--fits).
4. GPU-hours = dwarfs x (1 + mocks per dwarf) x [evaluations per fit x seconds per evaluation(N) + setup(N)]
   x overhead, for each field size in the budget file.

Usage: python dwarf/bench_dwarf_gpu.py --config dwarf/mock8/scl8mock.yaml --target dwarf/mock8/scl8mock_mock_target.fits \
           --fits dwarf/mock8/results_scl8mock/fit_gpu.json dwarf/mock8null/results_scl8null/fit_gpu.json \
           --budget dwarf/gpu_budget.yaml --out dwarf/results_bench
"""
import argparse, copy, csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import yaml
from fit_dwarf2 import load_field
from fit_dwarf_gpu import GPUModel, sync
from fit_dwarf import ell_radius

PER_STAR = ("xi", "eta", "BR", "G", "A", "Linv", "LinvA", "lp_ast_const", "fg_ns", "cmd_main", "lg")


def resized(m, N, rng):
    """Shallow copy of a GPUModel whose per-star tensors are resampled (with replacement) to N stars."""
    idx = torch.as_tensor(rng.integers(0, len(m.xi), N), device=m.dev)
    c = copy.copy(m)
    for k in PER_STAR:
        setattr(c, k, getattr(m, k)[idx])
    return c


def time_eval(m, th, repeats):
    for _ in range(3):  # warm-up
        th.grad = None; m.loss(th).backward()
    sync(m.dev)
    if m.dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(repeats):
        th.grad = None; m.loss(th).backward()
    sync(m.dev)
    mem = torch.cuda.max_memory_allocated() / 1e9 if m.dev.type == "cuda" else float("nan")
    return (time.time() - t0) / repeats, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--fits", nargs="+", required=True, help="fit_*.json from GPU fits (for evaluations per fit)")
    ap.add_argument("--budget", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=float, nargs="*", default=[1.5e5, 5e5, 1e6, 3e6, 1e7])
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = yaml.safe_load(open(a.config)); bud = yaml.safe_load(open(a.budget))
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    d, fgc = cfg["dwarf"], cfg["foreground"]
    lc = cfg["model"].get("logg_column", "logg_gspphot")
    tgt = load_field(a.target, (d["ra"], d["dec"]), 0.0, fgc["fit_radius_deg"], cfg["cuts"], lc)
    tmpl = load_field(a.target, (d["ra"], d["dec"]), fgc["fit_radius_deg"], d["radius_deg"], cfg["cuts"], lc)
    m = GPUModel(tgt, tmpl, cfg, 2, dev)
    r = ell_radius(tgt["xi"], tgt["eta"], d["ellipticity"], d["pa_deg"])
    m.set_hists((r < 3 * d["a_plummer_deg"]).astype(float))
    th = torch.tensor(m.theta, dtype=torch.float64, device=dev, requires_grad=True)
    rng = np.random.default_rng(0)
    rows = []
    for N in [int(s) for s in a.sizes]:
        try:
            s, mem = time_eval(resized(m, N, rng), th, a.repeats)
        except torch.cuda.OutOfMemoryError:
            print(f"N={N:>10,}: out of GPU memory, skipped"); torch.cuda.empty_cache(); continue
        rows.append(dict(N=N, sec_per_eval=s, peak_mem_gb=mem))
        print(f"N={N:>10,}  {s * 1e3:8.2f} ms per likelihood+gradient   peak memory {mem:.2f} GB", flush=True)
    with open(os.path.join(a.out, "dwarf_timings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # evaluations and setup time per full fit (1- + 2-component), from the real GPU fits
    F = [json.load(open(p)) for p in a.fits if os.path.exists(p)]
    evals = np.mean([f["models"]["1"]["n_evals"] + f["models"]["2"]["n_evals"] for f in F])
    setup = np.mean([f["seconds_setup"] for f in F]); n_fit = np.mean([f["n_fit"] for f in F])
    opt = np.mean([f["seconds_optimizer"] for f in F]); total = np.mean([f["seconds_total"] for f in F])
    Ns = np.array([r["N"] for r in rows], float); ts = np.array([r["sec_per_eval"] for r in rows])
    slope, icpt = np.polyfit(Ns, ts, 1)
    t_of = lambda N: np.interp(N, Ns, ts) if Ns.min() <= N <= Ns.max() else icpt + slope * N
    runs = bud["dwarfs"] * (1 + bud["mocks_per_dwarf"])
    table = []
    for N in bud["fields_per_dwarf"]:
        per_fit = evals * t_of(N) + setup * N / n_fit    # setup (foreground XD template + histograms) scaled ~ N
        table.append(dict(N=int(N), sec_per_eval=float(t_of(N)), sec_per_fit=float(per_fit),
                          gpu_hours=float(runs * per_fit * bud["overhead"] / 3600)))
    res = dict(gpu=gpu, timings=rows, evals_per_fit=float(evals), measured_fit_stars=float(n_fit),
               measured_fit_seconds_total=float(total), measured_optimizer_seconds=float(opt),
               measured_setup_seconds=float(setup), budget=bud, runs=runs, table=table,
               cpu_reference_seconds=[f.get("compare_to_cpu", {}).get("cpu_seconds") for f in F])
    json.dump(res, open(os.path.join(a.out, "dwarf_gpu_budget.json"), "w"), indent=2, default=float)
    warn = "WARNING: measured on CPU -- a pipeline test only, not usable for the proposal.\n\n" if gpu == "cpu" else ""
    md = [warn + f"Measured on: {gpu}. Dwarf-halo model (1- and 2-component fits, {cfg['model']['em_iterations']} EM steps each).",
          "",
          f"A full fit of a {n_fit:,.0f}-star field took {total:,.0f} s on this GPU ({evals:,.0f} likelihood+gradient "
          f"evaluations; setup incl. the foreground XD template {setup:,.0f} s).",
          "", "| Stars in field | ms per evaluation | s per full fit | GPU-h for the plan (x%.1f overhead) |" % bud["overhead"],
          "| --- | --- | --- | --- |"]
    for t in table:
        md.append(f"| {t['N']:,} | {t['sec_per_eval'] * 1e3:.2f} | {t['sec_per_fit']:,.0f} | {t['gpu_hours']:,.0f} |")
    md += ["", f"Plan: {bud['dwarfs']} dwarfs x (1 data + {bud['mocks_per_dwarf']} mocks) = {runs:,} full fits per field size.",
           f"Timed sizes: {', '.join(f'{int(n):,}' for n in Ns)} stars"
           + (" (larger field sizes extrapolated linearly)" if max(bud['fields_per_dwarf']) > Ns.max() else "") + ".",
           "Evaluations per fit are taken from the measured ~150k-star fits and assumed not to grow with N.",
           "No speed-up is assumed for the H200."]
    cref = [c for c in res["cpu_reference_seconds"] if c]
    if cref:
        md.append(f"Same fit on CPU (fit_dwarf2.py, finite-difference gradients): {np.mean(cref):,.0f} s.")
    open(os.path.join(a.out, "dwarf_gpu_budget.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
