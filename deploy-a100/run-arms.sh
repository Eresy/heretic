#!/bin/bash
# One arm, 400 trials, K=4 k-means directions. Run only after probe-memory.sh confirms
# it fits and gives a real s/trial.
#
# The 20-trial k1/groups arms are gone: at 20 trials neither produced a single trial
# under 10 refusals, so they measured nothing. The budget goes to one arm that can
# actually search the widened space (max_weight_position floor 0.2 rather than 0.6, and
# linear_attn.out_proj split off from attn.o_proj). Add `run groups
# "$DEPLOY/config-groups.toml"` below to spend another ~6 h on the named-topic arm.
set -e

# The vastai/pytorch image keeps its environment here.
source /venv/main/bin/activate
cd "$(dirname "$0")/.."
DEPLOY=$PWD/deploy-a100
MODEL=/workspace/Qwen3.8-27B
CONSOLE=/workspace/runs/console.log

# Logging is tmux's job, not a pipe's. `heretic ... | tee` used to cost two things: the
# pipeline's exit status became tee's, so `set -e` never saw heretic crash, and heretic's
# stdout was not a terminal, so rich rendered plain. pipe-pane taps the pane instead,
# leaving heretic on the pty.
mkdir -p /workspace/runs
if [ -n "${TMUX:-}" ]; then
    tmux pipe-pane -o "cat >> $CONSOLE"
    echo "console log: $CONSOLE"
else
    echo "NOTE: not inside tmux, so nothing is being logged to $CONSOLE." >&2
    echo "      Start with: tmux new -d -s run 'bash $0'" >&2
fi

run() {  # name, config, extra args...
    local name=$1 config=$2; shift 2
    echo "########## arm: $name ##########"
    mkdir -p "/workspace/runs/$name"
    cp "$config" "/workspace/runs/$name/config.toml"
    cd "/workspace/runs/$name"
    local started=$(date +%s)
    # --model must be explicit: absent from argv, main.py:325 inserts it before
    # the last argument and swallows that flag's value.
    heretic --model "$MODEL" --study-checkpoint-dir "checkpoints-$name" "$@"
    echo "ARM $name WALL CLOCK: $(( $(date +%s) - started ))s"
    cd - >/dev/null
}

run k4 "$DEPLOY/config.toml" --num-refusal-directions 4

echo
echo "Pareto fronts:"
grep -a "\[Trial" "$CONSOLE" || true
