#!/bin/bash
# Three arms, 20 trials each, identical except for how directions are obtained.
# Run only after probe-memory.sh confirms K=4 fits and gives a real s/trial.
set -e

# The vastai/pytorch image keeps its environment here.
source /venv/main/bin/activate
cd "$(dirname "$0")/.."
DEPLOY=$PWD/deploy-a100
MODEL=/workspace/Qwen3.8-27B

run() {  # name, config, extra args...
    local name=$1 config=$2; shift 2
    echo "########## arm: $name ##########"
    mkdir -p "/workspace/runs/$name"
    cp "$config" "/workspace/runs/$name/config.toml"
    cd "/workspace/runs/$name"
    local started=$(date +%s)
    # --model must be explicit: absent from argv, main.py:325 inserts it before
    # the last argument and swallows that flag's value.
    heretic --model "$MODEL" --study-checkpoint-dir "checkpoints-$name" "$@" \
        2>&1 | tee "$name.log"
    echo "ARM $name WALL CLOCK: $(( $(date +%s) - started ))s"
    cd - >/dev/null
}

run k1     "$DEPLOY/config.toml"        --num-refusal-directions 1
run k4     "$DEPLOY/config.toml"        --num-refusal-directions 4
run groups "$DEPLOY/config-groups.toml"

echo
echo "Pareto fronts:"
grep -h "\[Trial" /workspace/runs/*/*.log || true
