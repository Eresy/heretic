#!/bin/bash
# FIRST THING TO RUN. Answers two questions in ~15 minutes:
#
#   1. Does clustered K=4 fit in 80 GB of VRAM? Locally it used 105 GB of
#      UNIFIED memory, but that figure conflates VRAM with host RAM. On a
#      discrete card, offload_outputs_to_cpu sends residuals to system RAM, so
#      the VRAM requirement should be far lower -- this measures it instead of
#      guessing. If it OOMs, rerun with --batch-size 64.
#   2. What is the real seconds-per-trial here? Local was 211 s; everything
#      budgeted for this session is extrapolated from that and needs replacing
#      with a measurement.
set -e

# The vastai/pytorch image keeps its environment here.
source /venv/main/bin/activate
cd "$(dirname "$0")/.."
mkdir -p /workspace/runs/probe && cp deploy-a100/config.toml /workspace/runs/probe/

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -l 5 > /workspace/runs/probe/vram.csv &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

cd /workspace/runs/probe
started=$(date +%s)
heretic --model /workspace/Qwen3.8-27B \
        --num-refusal-directions 4 \
        --n-trials 2 --n-startup-trials 2 \
        --study-checkpoint-dir checkpoints-probe \
    2>&1 | tee probe.log
elapsed=$(( $(date +%s) - started ))

kill $SAMPLER 2>/dev/null
echo
echo "wall clock: ${elapsed}s"
echo "peak VRAM : $(sort -n /workspace/runs/probe/vram.csv | tail -1) MiB"
grep -E "Elapsed time" probe.log | tail -2
