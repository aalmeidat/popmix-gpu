#!/bin/bash
# Dwarf-halo model on the GPU: check the port against the CPU results, run the fixed model on the halo and
# null mocks, time it, and (if dwarf/sculptor8_target.fits is present) refit real Sculptor.
# Submitted by:  bash rivanna_start.sh <allocation> dwarf
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J rubin_dwarf_gpu
#SBATCH -o logs/dwarf_gpu-%j.out
#SBATCH -e logs/dwarf_gpu-%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh
nvidia-smi
M=dwarf/mock8; N=dwarf/mock8null
# the mock fields are generated on first use (the halo mock is reproduced exactly from the default seed)
[ -f $M/scl8mock_mock_target.fits ] || pyrun dwarf/make_mock_dwarf.py --config $M/scl8mock.yaml --fg_density 500
[ -f $N/scl8null_mock_target.fits ] || pyrun dwarf/make_mock_dwarf.py --config $N/scl8null.yaml --fg_density 500 --halo_frac 0
echo "== 1/5 port check: CPU-compatible mode on the halo mock, compared with the CPU run"
pyrun dwarf/fit_dwarf_gpu.py --config $M/scl8mock.yaml --target $M/scl8mock_mock_target.fits --compat \
      --reference $M/results_scl8mock/fit_v2.json
echo "== 2/5 fixed model (halo color shift fitted) on the halo mock"
pyrun dwarf/fit_dwarf_gpu.py --config $M/scl8mock.yaml --target $M/scl8mock_mock_target.fits \
      --reference $M/results_scl8mock/fit_v2.json
echo "== 3/5 fixed model on the null mock (no halo injected)"
pyrun dwarf/fit_dwarf_gpu.py --config $N/scl8null.yaml --target $N/scl8null_mock_target.fits
echo "== 3b  halo mock with a bluer (more metal-poor) halo: injected color shift -0.08 mag"
pyrun dwarf/make_mock_dwarf.py --config dwarf/mock8shift/scl8shift.yaml --halo_dcol -0.08 --fg_density 500
pyrun dwarf/fit_dwarf_gpu.py --config dwarf/mock8shift/scl8shift.yaml --target dwarf/mock8shift/scl8shift_mock_target.fits
echo "== 4/5 timing and GPU-hour estimate"
pyrun dwarf/bench_dwarf_gpu.py --config $M/scl8mock.yaml --target $M/scl8mock_mock_target.fits \
      --fits $M/results_scl8mock/fit_gpu.json $N/results_scl8null/fit_gpu.json dwarf/mock8shift/results_scl8shift/fit_gpu.json \
      --budget dwarf/gpu_budget.yaml --out dwarf/results_bench
echo "== 5/5 real Sculptor"
if [ -f dwarf/sculptor8_target.fits ]; then
  pyrun dwarf/fit_dwarf_gpu.py --config dwarf/sculptor8.yaml --plots \
    || echo "Sculptor fit failed (the mock results above are still valid)"
else
  echo "skipped: dwarf/sculptor8_target.fits not found (optional; fetch it with dwarf/pull_gaia_dwarf.py --config dwarf/sculptor8.yaml --xp)"
fi
echo
echo "================ PASTE EVERYTHING BELOW THIS LINE ================"
pyrun dwarf/summarize_gpu_job.py
