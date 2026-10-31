"""Short summary of the dwarf GPU job (the part to paste back)."""
import json, os

here = os.path.dirname(os.path.abspath(__file__))
J = lambda *p: json.load(open(os.path.join(here, *p))) if os.path.exists(os.path.join(here, *p)) else None


def line(tag, f):
    if f is None:
        print(f"{tag}: not run"); return
    m2 = f["models"]["2"]
    print(f"{tag}: n={f['n_fit']:,}  dlnL(2-1)={f['delta_lnL_2_minus_1']:.1f}  dBIC(1-2)={f['delta_BIC_1_minus_2']:.1f}  "
          f"f_halo={m2['halo_fraction_of_dwarf']:.3f}  a_h={m2['a_halo_arcmin']:.1f}'  at_bound={m2['at_bound']}  "
          f"dcol={m2['halo_color_shift']:+.3f}  time={f['seconds_total']:.0f}s on {f['gpu']}")
    c = f.get("compare_to_cpu")
    if c and f.get("mode") == "compat":   # linear mode uses a smoother density, so its lnL is not comparable
        print("   vs CPU: " + "; ".join(f"K={k} dlnL(gpu-cpu)={v['dlnL_gpu_minus_cpu']:+.2f}" for k, v in c["models"].items())
              + f"; CPU took {c['cpu_seconds']:.0f}s")


line("port check (compat), halo mock", J("mock8", "results_scl8mock", "fit_gpu_compat.json"))
line("fixed model, halo mock (injected 7.9% / 40')", J("mock8", "results_scl8mock", "fit_gpu.json"))
line("fixed model, null mock", J("mock8null", "results_scl8null", "fit_gpu.json"))
line("fixed model, bluer-halo mock (injected dcol -0.08)", J("mock8shift", "results_scl8shift", "fit_gpu.json"))
line("real Sculptor", J("results_sculptor8", "fit_gpu.json"))
p = os.path.join(here, "results_bench", "dwarf_gpu_budget.md")
print()
print(open(p).read() if os.path.exists(p) else "timing: not run")
