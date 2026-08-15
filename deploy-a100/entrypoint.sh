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

python -c "
import torch, sys
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit('FATAL: torch cannot see the GPU -- a CPU build would run silently at a crawl')
p = torch.cuda.get_device_properties(0)
print('device', p.name, f'{p.total_memory/1024**3:.0f} GB')
"

echo "=== topic sets ==="
# Shipped in the repo (212 KB), so the clone brings them and nothing needs
# uploading before onstart runs.
mkdir -p /workspace/data
tar xzf "$(dirname "$0")/topic-sets.tgz" -C /workspace/data
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
