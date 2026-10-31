"""Train the deconvolving normalizing flow (and optionally a plain flow for comparison).

Outputs (in output_dir/flow/):
  flow_<tag>.pt            weights + scaler
  flow_<tag>_samples.npy   deconvolved samples in physical units (for plotting)
  flow_<tag>_summary.json  losses, timing, GPU name
Usage:
  python scripts/fit_flow.py --config config.yaml            # deconvolving flow
  python scripts/fit_flow.py --config config.yaml --plain    # ordinary flow on noisy data
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from proto.data import load_config, load_catalog, Scaler, save_json
from proto.flow import make_flow, deconv_nll, cholesky_safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--plain", action="store_true", help="no deconvolution (baseline)")
    ap.add_argument("--quick", action="store_true", help="few epochs, for testing")
    a = ap.parse_args()
    cfg = load_config(a.config)
    c = cfg["flow"]
    if a.quick:
        c = dict(c, max_epochs=25, patience=8, n_samples_out=20000)
    tag = "plain" if a.plain else "deconv"
    out = os.path.join(cfg["output_dir"], "flow")
    os.makedirs(out, exist_ok=True)

    if cfg["catalog"].get("missing", "drop") != "drop":
        print("note: the flow uses only stars complete in every dimension; "
              "marginalizing over missing dimensions is part of the proposed research.")
        cfg = dict(cfg, catalog=dict(cfg["catalog"], missing="drop"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])

    cat = load_catalog(cfg)
    sc = Scaler().fit(cat["X"])
    Z = torch.as_tensor(sc.x(cat["X"]), dtype=torch.float32)
    L = cholesky_safe(torch.as_tensor(sc.cov(cat["S"]), dtype=torch.float32))
    n, D = Z.shape
    perm = rng.permutation(n)
    nv = max(1, int(c["val_fraction"] * n))
    iv, it = perm[:nv], perm[nv:]
    Ztr, Ltr, Zva, Lva = Z[it].to(dev), L[it].to(dev), Z[iv].to(dev), L[iv].to(dev)

    flow = make_flow(D, c["transforms"], c["hidden"]).to(dev)
    opt = torch.optim.Adam(flow.parameters(), lr=c["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, c["patience"] // 3))
    best_val, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    steps = 0
    for ep in range(c["max_epochs"]):
        flow.train()
        idx = torch.randperm(len(Ztr), device=dev)
        tl = []
        for b in range(0, len(idx), c["batch_size"]):
            j = idx[b:b + c["batch_size"]]
            loss = deconv_nll(flow, Ztr[j], Ltr[j], c["mc_samples"], deconvolve=not a.plain)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(loss.item())
            steps += 1
        flow.eval()
        with torch.no_grad():
            vl = np.mean([deconv_nll(flow, Zva[b:b + 4096], Lva[b:b + 4096], c["mc_samples_val"],
                                     deconvolve=not a.plain).item() for b in range(0, len(Zva), 4096)])
        sched.step(vl)
        hist.append(dict(epoch=ep, train=float(np.mean(tl)), val=float(vl)))
        if ep % 10 == 0:
            print(f"epoch {ep}: train {np.mean(tl):.4f}  val {vl:.4f}")
        if vl < best_val - 1e-4:
            best_val, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.time() - t0
    flow.load_state_dict(best_state)
    with torch.no_grad():
        smp = flow().sample((c["n_samples_out"],)).cpu().numpy()
    smp = sc.inv_x(smp)
    torch.save(dict(state=best_state, scaler=sc.to_dict(), cfg=c, dims=cat["names"]),
               os.path.join(out, f"flow_{tag}.pt"))
    np.save(os.path.join(out, f"flow_{tag}_samples.npy"), smp)
    save_json(dict(device=gpu, tag=tag, n_train=len(it), n_val=nv, epochs=len(hist),
                   optimizer_steps=steps, train_seconds=train_s,
                   seconds_per_step=train_s / max(steps, 1), best_val_nll=float(best_val),
                   mc_samples=c["mc_samples"] if not a.plain else 0, history=hist),
              os.path.join(out, f"flow_{tag}_summary.json"))
    print(f"[{tag}] best val NLL {best_val:.4f} after {len(hist)} epochs, {train_s:.1f}s on {gpu}")


if __name__ == "__main__":
    main()
