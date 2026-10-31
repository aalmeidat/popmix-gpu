#!/bin/bash
# One GPU job that produces the proposal's GPU-hour number. Submitted by rivanna_start.sh.
# It uses the 16k-star mock (timings depend on star count, dimensions and K, not on which
# stars they are), so no SDSS-V data has to go to Rivanna.
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J rubin_gpu_bench
#SBATCH -o logs/gpu_bench-%j.out
#SBATCH -e logs/gpu_bench-%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh
nvidia-smi
echo "== 1/3 make the mock catalog"
pyrun scripts/make_mock.py --config config_mock.yaml
echo "== 2/3 quick mixture fit (gives the model the benchmark samples from)"
pyrun scripts/fit_xd.py --config config_mock.yaml --quick
echo "== 3/3 timings from 10^4 to 10^6 stars and the GPU-hour extrapolation"
pyrun scripts/benchmark.py --config config_mock.yaml
echo "== done. The number for the proposal:"
cat results_mock/benchmark/gpu_budget.md
