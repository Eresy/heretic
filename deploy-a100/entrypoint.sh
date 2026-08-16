#!/bin/bash
# One-time instance setup. Expects HF_TOKEN in the environment, passed via the
# template's env or `vastai create instance --env`, never hardcoded.
#
# Usable as a template onstart script. Note that onstart runs once the container
# is `running`, which is when GPU billing starts -- there is no way to do the
# download in the storage-only window. It is ~1-4 min for 52 GB over Xet, so
# roughly $0.05 of GPU time. To prep now and run later, `vastai stop instance`
# afterwards: stopped instances bill storage only and keep the disk.
set -e

# The vastai/pytorch image keeps its environment here. Without this, pip installs
# land somewhere the run will not see, and torch may get reinstalled from PyPI
# over the image's CUDA build.
source /venv/main/bin/activate

# The template's env reaches onstart, but NOT a later non-interactive shell -- so
# `ssh <host> bash entrypoint.sh` or any re-run sees an empty HF_TOKEN and the model
# download fails on a gated repo. Persist it on the first pass, read it back on later
# ones. /etc/environment takes KEY=value lines, which is why only this one variable is
# written rather than the whole of `env` (LS_COLORS and friends contain characters that
# break the file's parser).
if [ -z "${HF_TOKEN:-}" ] && [ -r /etc/environment ]; then
    HF_TOKEN=$(sed -n 's/^HF_TOKEN=//p' /etc/environment | tail -1)
    export HF_TOKEN
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: HF_TOKEN is empty. Qwen3.8-27B is gated, so the download would 401." >&2
    exit 1
fi

grep -q '^HF_TOKEN=' /etc/environment 2>/dev/null || echo "HF_TOKEN=$HF_TOKEN" >> /etc/environment

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== heretic ==="
cd /workspace
[ -d heretic ] || git clone -b feat/topic-direction-groups \
    https://github.com/Eresy/heretic.git
cd heretic
git pull --ff-only || true

# torch is already present and correct for this machine. `-e .` leaves it alone
# because pyproject deliberately does not pin a torch version.
pip install -q -e '.[research]'

echo "=== GatedDeltaNet fast path ==="
# transformers gates the GatedDeltaNet fast path on BOTH causal-conv1d and
# flash-linear-attention being importable (modeling_qwen3_5.py, is_fast_path_available).
# With either missing it silently uses torch_chunk_gated_delta_rule, which is what the
# first A100 run did -- and heretic calls transformers.logging.set_verbosity_error(), so
# the warning never reaches the log and there is no way to notice from the output.
#
# flash-linear-attention is pure Python plus Triton, so PyPI has a wheel.
# causal-conv1d publishes only an sdist to PyPI; the prebuilt wheels live on GitHub
# releases and are tagged by exact torch minor, CUDA major, cpython version and C++ ABI.
# That is why the image tag must be pinned: with @vastai-automatic-tag the torch version
# is whatever the machine resolves to, no wheel matches, and pip compiles with nvcc for
# 30-60 minutes of GPU-billed time.
pip install -q flash-linear-attention

# transformers is PINNED, and the pin is load-bearing. 5.15 renamed the kernel it looks
# for from `fused_recurrent_gated_delta_rule` to `recurrent_gated_delta_rule`, ahead of
# any fla release exporting that name -- fla 0.5.2 is the latest on PyPI and still exports
# the old one. The lookup therefore returns None and the RECURRENT path silently falls
# back to torch, which is the per-token decode loop and the bulk of the wall clock. The
# chunk (prefill) and causal-conv1d kernels bind either way, so the failure is invisible
# unless you check each one. Measured on 5.15: 3 of 4 bound.
pip install -q "transformers==5.14.1"

CONV1D_VERSION="1.6.2.post1"
TORCH_VERSION="$(python -c 'import torch; print(".".join(torch.__version__.split(".")[:2]))')"
CUDA_MAJOR="cu$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')"
ABI="$(python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
PYTHON_TAG="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
WHEEL="causal_conv1d-${CONV1D_VERSION}+${CUDA_MAJOR}torch${TORCH_VERSION}cxx11abi${ABI}-${PYTHON_TAG}-${PYTHON_TAG}-linux_x86_64.whl"
echo "wheel: $WHEEL"

# --no-build-isolation is not enough to prevent a source build; installing the wheel by
# URL is. If the URL 404s the pin is wrong, and stopping is correct -- a silent fallback
# to the sdist would burn the build time this whole block exists to avoid.
pip install -q "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CONV1D_VERSION}/${WHEEL}"

# Verify the OUTCOME, not the precondition. An earlier version of this check only proved
# that causal_conv1d and fla import, which they do even when transformers then fails to
# bind one of them -- exactly the 5.15 case above. On the pinned 5.14 the module-level
# `is_fast_path_available` is authoritative: it is the AND of all four kernels having been
# imported successfully. Treat its absence as fatal, since that means transformers has
# moved to the decorator mechanism and this check no longer proves anything.
python -c "
import sys

import transformers
import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

available = getattr(modeling, 'is_fast_path_available', None)

if available is None:
    sys.exit(
        f'FATAL: transformers {transformers.__version__} has no is_fast_path_available. '
        'The pin slipped, and kernel binding can no longer be verified this way.'
    )

if not available:
    missing = [
        name
        for name in (
            'causal_conv1d_fn',
            'causal_conv1d_update',
            'chunk_gated_delta_rule',
            'fused_recurrent_gated_delta_rule',
        )
        if getattr(modeling, name, None) is None
    ]
    sys.exit(f'FATAL: GatedDeltaNet fast path off, unbound kernels: {missing}')

print(f'fast path fully bound (transformers {transformers.__version__})')
"

python -c "
import torch, sys
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit('FATAL: torch cannot see the GPU -- a CPU build would run silently at a crawl')
p = torch.cuda.get_device_properties(0)
print('device', p.name, f'{p.total_memory/1024**3:.0f} GB')
"

echo "=== topic sets ==="
# Shipped in the repo as JSONL, not a tarball: this repo's .gitattributes is
# `* text eol=lf`, which line-ending-normalises binaries and silently corrupts
# them. JSON escapes the newlines that 10 of the 3673 prompts contain, so the
# sets survive git intact and datasets reads them directly.
mkdir -p /workspace/data
cp -r "$(dirname "$0")/topic-sets/." /workspace/data/
ls /workspace/data | tr '\n' ' '; echo

echo "=== base model ==="
# Qwen3.8-27B is Xet-backed, so hf_xet handles the transfer and supersedes
# hf_transfer. Setting HF_HUB_ENABLE_HF_TRANSFER here would simply be ignored.
pip install -q "huggingface_hub[hf_xet]"
export HF_XET_HIGH_PERFORMANCE=1
hf download Qwen/Qwen3.8-27B --local-dir /workspace/Qwen3.8-27B --token "$HF_TOKEN"
du -sh /workspace/Qwen3.8-27B
ls /workspace/Qwen3.8-27B/*.safetensors | wc -l | xargs -I{} echo "{} shards"

echo
echo "Setup complete. Next: bash /workspace/heretic/deploy-a100/probe-memory.sh"
