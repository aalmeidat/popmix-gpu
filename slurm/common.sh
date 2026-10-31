# Shared settings for the Slurm scripts (sourced, not run).
#
# Finds UVA RC's PyTorch container automatically: loads the apptainer + pytorch modules and
# picks the newest pytorch-*.sif in $CONTAINERDIR. The choice is saved in .pytorch_sif so the
# batch jobs use the same container as the setup step. To force a version:
#   PYTORCH_SIF=$CONTAINERDIR/pytorch-2.0.1.sif bash rivanna_start.sh <allocation>

module purge
module load apptainer pytorch 2>/dev/null || module load apptainer

_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -z "$PYTORCH_SIF" ] && [ -f "$_root/.pytorch_sif" ]; then
  PYTORCH_SIF=$(cat "$_root/.pytorch_sif")
fi
if [ -z "$PYTORCH_SIF" ]; then
  PYTORCH_SIF=$(ls -1 "$CONTAINERDIR"/pytorch-*.sif 2>/dev/null | sort -V | tail -1)
fi
if [ -z "$PYTORCH_SIF" ] || [ ! -f "$PYTORCH_SIF" ]; then
  echo "Could not find a PyTorch container (CONTAINERDIR='$CONTAINERDIR')."
  echo "Run 'module spider pytorch' to see the versions, then rerun with"
  echo "  PYTORCH_SIF=\$CONTAINERDIR/pytorch-<version>.sif bash rivanna_start.sh <allocation>"
  return 1 2>/dev/null || exit 1
fi
echo "$PYTORCH_SIF" > "$_root/.pytorch_sif"
export PYTORCH_SIF

# Run a Python script inside the container with GPU access.
pyrun() { apptainer exec --nv "$PYTORCH_SIF" python "$@"; }
