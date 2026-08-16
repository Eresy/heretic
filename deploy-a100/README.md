# Qwen3.8-27B abliteration on a rented A100

Everything here encodes measurements taken locally on Qwen3.8-27B. None of it is
inherited from `config.default.toml`, whose defaults are actively wrong for this
model — `row_normalization = "full"` plus `orthogonalize_direction = true`
produces an abliteration that does nothing, at a suspiciously beautiful KL.

## Session

```bash
# local
vastai create ssh-key ~/.ssh/id_ed25519.pub
vastai create instance <OFFER_ID> \
    --template_hash 0c6e160add3aaf47aa29db36d60c815b \
    --disk 300 --ssh --direct --label heretic-qwen38
vastai show instance <ID> --raw          # poll until actual_status == "running"
# actual_status of exited/unknown/offline will NEVER reach running -- destroy and
# pick another offer rather than looping while storage bills.

# kitty's TERM is not in the remote terminfo and breaks curses rendering. This is
# needed on EVERY connection, not just the first.
TERM=xterm ssh -p <PORT> root@<HOST>        # or: TERM=xterm $(vastai ssh-url <ID>)
# on the box -- onstart already ran entrypoint.sh; check it finished
tail -20 /var/log/onstart.log 2>/dev/null || bash /workspace/heretic/deploy-a100/entrypoint.sh
bash heretic/deploy-a100/probe-memory.sh      # ~15 min, RUN THIS FIRST
bash heretic/deploy-a100/run-arms.sh          # 1 arm x 400 trials, ~6 h

# copy results out, then destroy IMMEDIATELY — do not analyse while it bills
vastai copy <ID>:/workspace/runs local:./runs
vastai destroy instance <ID> -y
```

## Template

`2bba4750eab98876561b62c3d9ca80d9` **as of 2026-08-16 — expect this to be stale.**
`vastai update template` mints a NEW `hash_id` on every call, so any hash written down
here is only correct until the next edit. There is no `vastai show template`, and the
REST `?hash_id=` filter is ignored; look up the current one with
`GET /api/v0/users/current/templates/` (bearer key from
`~/.config/vastai/vast_api_key`). That response contains the HF token in `env` — redact
before pasting it anywhere.

Already applied; kept here as the recipe.

```bash
vastai update template <CURRENT_HASH> \
    --image vastai/pytorch --image_tag 2.10.0-cuda-12.8.1-py313-24.04-2026-06-15 \
    --disk_space 300 --ssh --direct \
    --env '-e HF_TOKEN=<token> -e HF_XET_HIGH_PERFORMANCE=1' \
    --onstart-cmd 'git clone -b feat/topic-direction-groups https://github.com/Eresy/heretic.git /workspace/heretic && bash /workspace/heretic/deploy-a100/entrypoint.sh'
```

The onstart has to clone first: it cannot reference a path inside the repo that
does not exist yet. The topic sets ride along in the repo (212 KB), so nothing
needs uploading before onstart runs -- `vastai copy` only works once the instance
is already up, which is after onstart.

- **disk 200 -> 300 GB**: 52 GB model, plus HF cache, plus a 52 GB merged export.
- **the image tag must be PINNED, not `@vastai-automatic-tag`.** The automatic tag
  resolves torch to whatever suits the machine, and `causal-conv1d` ships prebuilt
  wheels only for exact (CUDA major, torch minor, cpython, C++ ABI) tuples. With an
  unpinned torch nothing matches and pip falls back to compiling with nvcc — 30-60
  minutes of GPU-billed build. That is what happened on the first run.
  `2.10.0-cuda-12.8.1-py313-24.04-2026-06-15` matches
  `causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl`,
  which is verified to exist. Pinning the CUDA build means the host has to support it,
  so add `cuda_max_good>=12.8` to the offer search.

  **torch 2.10 is the ceiling, not a conservative choice.** The full x86_64 matrix for
  causal-conv1d 1.6.2.post1 (the current release) is cu11 torch2.6-2.7, cu12
  torch2.6-2.10, cu13 torch2.9-2.10, all cp310-cp313. `vastai/pytorch` also publishes
  2.11 and 2.12 — no wheels for either. The `cu13 torch25.11`..`torch26.04` rows look
  newer but are NGC container versions, cp312 only. Two ways the automatic tag loses,
  then: a torch above 2.10, or **py314**, which vastai builds and causal-conv1d does not.
- **env**: Xet needs the token. Do NOT also set `HF_HUB_ENABLE_HF_TRANSFER` --
  Qwen3.8-27B is Xet-backed (`xetEnabled: true`), so `hf_xet` handles the
  transfer and the hf_transfer flag is ignored.

On billing: `onstart` runs once the container is `running`, which is when GPU
charges start, so a download there is GPU-billed -- there is no way to use the
storage-only window. It is 1-4 minutes over Xet, about $0.05. To prep now and run
later, `vastai stop instance` afterwards: stopped instances bill storage only and
keep the disk.

## Why probe-memory.sh comes first

It answers the two things this plan is guessing about:

1. **Does clustered K=4 fit in 80 GB?** Locally it used 105 GB, but that was
   unified memory, which conflates VRAM with host RAM. On a discrete card
   `offload_outputs_to_cpu` puts residuals in system RAM, so the VRAM figure
   should be much lower. If it OOMs, add `--batch-size 64`; the transient scales
   linearly with batch. The grouped arm is unaffected either way — it needs only
   group means and runs at K=1 memory.
2. **What is the real seconds-per-trial?** Local was 211 s and every time
   estimate for this session is extrapolated from it. Replace the estimate with
   the measurement before committing to the long run.

## The arm

One arm: `k4`, 400 trials, K=4 k-means over bad residuals. The previous session ran
`k1`, `k4` and `groups` at 20 trials each; at that budget neither `k1` nor `groups`
produced a single trial under 10 refusals, so they measured nothing. The budget now goes
to one arm that can search the widened space — `max_weight_position` floored at
`0.2 * last_layer_index` rather than `0.6`, and `linear_attn.out_proj` optimized
separately from `attn.o_proj`. 400 trials also settles whether the 200-trial run merely
stopped early: its Pareto front was still improving at trial 193.

The `groups` arm is still the one no other heretic fork has, and is one line in
`run-arms.sh` away at roughly another 6 hours. The topic map measured pairwise
collinearity across the six topics (noise floor 0.988–0.993): drugs, chemistry, weapons,
infrastructure and hacking sit at 0.886–0.960 — one refusal mode — while NSFW stands
apart at 0.687–0.818. So K is 2 by measurement, each direction is identifiable, and four
directions that k-means would have spent on one mode are not spent. Fewer directions
removed means less capability removed.

## Settings that are not negotiable

- **Fit and score at the answer position** (`enable_thinking = false`). Fitting
  at the reasoning position gives 81–86/100 refusals against 44/100 with the same
  parameters. This was recovered from trohrbaugh's released weights: his
  direction is 0.87–0.92 collinear with the thinking-off direction, ~0.25 with
  any thinking-on variant.
- **`row_normalization = "none"` + `orthogonalize_direction = true`.** From the
  2×2 against an 88/100 baseline: none/true 44, full/true 52, full/false 64,
  none/false 69. Orthogonalization helps here, the opposite of gemma-3-12b,
  which the low S(g,r) (0.18–0.27 vs gemma's 0.957–0.985) predicted.
- **`checkpoint_action`, `trial_index`, `model_action` all pinned.** Missing
  `trial_index` leaves heretic blocked on an interactive Pareto menu reading a
  pipe: one CPU thread spinning, GPU at 0%, no error in the log, billing
  throughout. This cost 50 minutes locally.
- **The GatedDeltaNet fast path must be verified, not assumed.** `transformers` gates
  it on `causal-conv1d` *and* `flash-linear-attention` both being importable, and
  `main.py:434` calls `transformers.logging.set_verbosity_error()`, which swallows the
  warning it would otherwise print. The first run installed neither and spent 200 trials
  on `torch_chunk_gated_delta_rule` without a single line in the log to say so.
  `entrypoint.sh` now asserts `is_fast_path_available` and stops if it is false.
- **`--model` explicit on every invocation.** With it absent from argv,
  `main.py:325` inserts it before the last argument and swallows that flag's
  value — it silently ate `--study-checkpoint-dir` once already.

## Data

`topic-sets/` holds six topics as `{topic}-{fit,val,test}/train.jsonl` — 18 files, 3673
prompts — built only from
published benchmarks (SALAD-Bench, HarmBench, JBB-Behaviors, SORRY-Bench base
style), deduplicated across sources and filtered against
`mlabonne/harmful_behaviors` so the fitting sets cannot contaminate the eval.
`val` is what the search scores against; `test` is untouched until the final
model, so the val-to-test gap measures how much the search overfitted.

JSONL rather than parquet or a tarball: this repo's `.gitattributes` is
`* text eol=lf`, which normalises line endings in *binary* files too and
corrupts them. JSON escaping also preserves the embedded newlines that 10 of the
3673 prompts contain, which a plain one-per-line text file would silently split
into extra prompts.
