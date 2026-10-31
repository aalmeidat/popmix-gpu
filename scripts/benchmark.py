"""Time XD and the deconvolving flow at increasing N, then extrapolate GPU-hours for the proposal.

Synthetic catalogs are drawn from the fitted XD model, with measurement covariances
resampled from the real catalog, so the timing reflects realistic error structure.
Run fit_xd.py first.

Outputs (in output_dir/benchmark/):
  timings.csv, gpu_budget.json, gpu_budget.md (a paragraph you can adapt for the proposal)
Usage: python scripts/benchmark.py --config config.yaml
"""
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from proto.data import load_config, load_catalog, Scaler, save_json
from proto import xd
from proto.flow import make_flow, deconv_nll, cholesky_safe


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sizes", type=float, nargs="*", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    bc, fc, xc, bud = cfg["benchmark"], cfg["flow"], cfg["xd"], cfg["budget"]
    sizes = [int(s) for s in (a.sizes or bc["sizes"])]
    out = os.path.join(cfg["output_dir"], "benchmark")
    os.makedirs(out, exist_ok=True)
    dev = xd.device_auto()
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"

    m = np.load(os.path.join(cfg["output_dir"], "xd", "xd_best.npz"))
    alpha, mu, V = m["alpha"], m["mu"], m["V"]
    K, D = mu.shape
    cat = load_catalog(cfg)
    sc = Scaler(m["scaler_mean"], m["scaler_std"])
    SZ_real = sc.cov(cat["S"])
    rng = np.random.default_rng(0)

    rows = []
    for N in sizes:
        truth, _ = xd.sample(alpha, mu, V, N, rng)
        SZ = SZ_real[rng.integers(0, len(SZ_real), N)]
        Lc = np.linalg.cholesky(SZ + 1e-9 * np.eye(D))
        Z = truth + np.einsum("nij,nj->ni", Lc, rng.normal(size=(N, D)))
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # XD: fixed number of EM iterations
        t_xd = []
        for r in range(bc["repeats"]):
            f = xd.fit(Z, SZ, K, max_iter=bc["em_iters"], tol=-1, reg=xc["reg"], device=dev,
                       chunk=xc["chunk"], init=(alpha, mu, V))
            t_xd.append(f["seconds"] / f["n_iter"])
        mem_xd = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
        # flow: fixed number of optimizer steps at the configured batch size and M
        flow = make_flow(D, fc["transforms"], fc["hidden"]).to(dev)
        opt = torch.optim.Adam(flow.parameters(), lr=fc["lr"])
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
        Lt = cholesky_safe(torch.as_tensor(SZ, dtype=torch.float32, device=dev))
        t_fl = []
        for r in range(bc["repeats"]):
            sync(dev)
            t0 = time.time()
            for s in range(bc["flow_steps"]):
                j = torch.randint(0, N, (fc["batch_size"],), device=dev)
                loss = deconv_nll(flow, Zt[j], Lt[j], fc["mc_samples"])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sync(dev)
            t_fl.append((time.time() - t0) / bc["flow_steps"])
        row = dict(N=N, xd_s_per_iter=float(np.median(t_xd)), flow_s_per_step=float(np.median(t_fl)),
                   peak_gpu_mem_GB=mem_xd)
        rows.append(row)
        print(f"N={N:>9d}  XD {row['xd_s_per_iter']:.4f} s/iter   flow {row['flow_s_per_step']:.4f} s/step")
        del Zt, Lt, flow, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(out, "timings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # linear scaling t(N) = a + b N for XD; flow cost per step is ~independent of N
    Ns = np.array([r["N"] for r in rows], float)
    tx = np.array([r["xd_s_per_iter"] for r in rows])
    b, a0 = np.polyfit(Ns, tx, 1) if len(Ns) > 1 else (tx[0] / Ns[0], 0.0)
    a0 = max(a0, 0.0)
    t_step = float(np.median([r["flow_s_per_step"] for r in rows]))
    est = []
    for N in bud["target_N"]:
        t_iter = a0 + b * N
        xd_h = t_iter * bud["xd_iters"] * (bud["xd_K_values"] * bud["xd_inits"] + bud["xd_bootstrap"]) / 3600
        steps = bud["flow_epochs"] * np.ceil(N / fc["batch_size"])
        fl_h = t_step * steps * bud["flow_models"] / 3600
        tot = (xd_h + fl_h) * bud["overhead"]
        est.append(dict(N=N, xd_s_per_iter=t_iter, xd_gpu_hours=xd_h, flow_gpu_hours=fl_h,
                        total_gpu_hours_with_overhead=tot))
    save_json(dict(device=gpu, measured=rows, fit=dict(a=a0, b=b), flow_s_per_step=t_step,
                   assumptions=bud, flow_config=dict(batch_size=fc["batch_size"], mc_samples=fc["mc_samples"],
                                                     transforms=fc["transforms"], hidden=fc["hidden"]),
                   estimates=est), os.path.join(out, "gpu_budget.json"))

    unit = "GPU-h" if dev.type == "cuda" else "CPU-h"
    lines = [f"Measured on: {gpu}. Sizes timed: {', '.join(f'{int(n):,}' for n in Ns)} stars.", "",
             "| Stars | XD s/iter | XD {u} | Flow {u} | Total {u} (x{o:.1f} overhead) |".format(u=unit, o=bud["overhead"]),
             "| --- | --- | --- | --- | --- |"]
    for e in est:
        lines.append(f"| {int(e['N']):,} | {e['xd_s_per_iter']:.3f} | {e['xd_gpu_hours']:.1f} | "
                     f"{e['flow_gpu_hours']:.1f} | {e['total_gpu_hours_with_overhead']:.1f} |")
    lines += ["", "Assumptions: " + ", ".join(f"{k}={v}" for k, v in bud.items()),
              f"Flow: batch {fc['batch_size']}, {fc['mc_samples']} noise draws per star, "
              f"{fc['transforms']} spline layers of {fc['hidden']}.",
              "Covers Aim 1 only (mixture + flow). Simulation and rewinding costs need their own estimate.",
              "Times are for the GPU above; no speed-up is assumed for the H200."]
    if dev.type != "cuda":
        lines.insert(0, "WARNING: measured on CPU -- a pipeline test only, not usable for the proposal.\n")
    with open(os.path.join(out, "gpu_budget.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
