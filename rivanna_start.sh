#!/bin/bash
# Everything needed on Rivanna, in one command. Run it on a login node from this folder:
#
#   bash rivanna_start.sh                          # no argument: lists your allocations and stops
#   bash rivanna_start.sh <allocation>             # mixture model + flow benchmark  (logs/gpu_bench-<id>.out)
#   bash rivanna_start.sh <allocation> dwarf       # dwarf-halo model on the GPU     (logs/dwarf_gpu-<id>.out)
#
# Asks for an A100 by default (the newest PyTorch container has no kernels for the older V100s).
# GPU=any lets Slurm pick any GPU; GPU=<type> asks for another type.
set -e
cd "$(dirname "$0")"

ALLOC="$1"; JOB="${2:-bench}"
if [ -z "$ALLOC" ]; then
  echo "Allocations you can charge (the name in the first column is what goes after the script):"
  allocations 2>/dev/null || echo "  ('allocations' did not run; ask Nitya or Steve for the group's allocation name)"
  echo
  echo "Then run:  bash rivanna_start.sh <allocation_name> [dwarf]"
  exit 1
fi
case "$JOB" in
  bench) SCRIPT=slurm/gpu_benchmark.sbatch; LOG=gpu_bench ;;
  dwarf) SCRIPT=slurm/dwarf_gpu.sbatch; LOG=dwarf_gpu ;;
  *) echo "unknown job '$JOB' (use 'dwarf' or leave it out)"; exit 1 ;;
esac

source slurm/common.sh
echo "container: $PYTORCH_SIF"

echo "checking packages (zuko, astropy, pyyaml, scipy, matplotlib) ..."
apptainer exec "$PYTORCH_SIF" python -c "import zuko, astropy, yaml, scipy, matplotlib" 2>/dev/null || \
  apptainer exec "$PYTORCH_SIF" python -m pip install --user --quiet --no-warn-script-location zuko astropy pyyaml scipy matplotlib
apptainer exec "$PYTORCH_SIF" python -c "import torch, zuko, astropy, yaml, scipy, matplotlib; print('packages ok | torch', torch.__version__)"

mkdir -p logs
GPU="${GPU:-a100}"
GRES=()
[ "$GPU" != "any" ] && GRES=(--gres=gpu:$GPU:1)
jid=$(sbatch --parsable -A "$ALLOC" "${GRES[@]}" "$SCRIPT")
echo
echo "submitted GPU job $jid ($JOB)."
echo "  is it running?     squeue -u \$USER"
echo "  follow the output: tail -f logs/$LOG-$jid.out"
echo "  when it finishes:  the end of logs/$LOG-$jid.out has the part to paste back"
