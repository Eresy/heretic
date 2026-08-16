# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026  Marco Bucchiarone

"""
Reopens heretic's Pareto menu on a finished study so a trial can be chosen by hand.

An unattended run needs `trial_index` and `model_action` pinned, or it blocks forever on
an interactive prompt reading a pipe. That pinning is also what stops the menu appearing
at the end, and `--checkpoint-action continue` cannot undo it: main.py:538 replaces the
whole settings object with the copy stored in the study, so CLI flags and edits to
config.toml are both discarded.

The stored copy is a study user attribute in the Optuna journal, and the journal is an
append-only log whose later records win. So appending one corrected record is enough --
no trial has to be re-run, and nothing has to be rebuilt from the log by hand.

    python select-trial.py checkpoints-k4/--workspace--Qwen3--8-27B.jsonl
    heretic --model /workspace/Qwen3.8-27B \
            --study-checkpoint-dir checkpoints-k4 --checkpoint-action continue
"""

import json
import sys
from pathlib import Path

# Every field that suppresses a prompt we now want to see.
UNSET = ("trial_index", "model_action", "export_strategy")

# Optuna's journal operation code for "set study user attribute".
SET_STUDY_USER_ATTR = 2


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    journal = Path(sys.argv[1])
    records = [json.loads(line) for line in journal.read_text().splitlines() if line]

    settings_records = [
        record
        for record in records
        if record.get("op_code") == SET_STUDY_USER_ATTR
        and "settings" in (record.get("user_attr") or {})
    ]
    if not settings_records:
        print(f"No stored settings found in {journal}", file=sys.stderr)
        return 1

    latest = settings_records[-1]
    settings = json.loads(latest["user_attr"]["settings"])

    already = [field for field in UNSET if settings.get(field) is None]
    if len(already) == len(UNSET):
        print("Already patched -- heretic will prompt. Nothing to do.")
        return 0

    for field in UNSET:
        print(f"  {field}: {settings.get(field)!r} -> None")
        settings[field] = None

    record = dict(latest)
    record["user_attr"] = {"settings": json.dumps(settings)}
    with journal.open("a") as file:
        file.write(json.dumps(record) + "\n")

    print(f"Appended a corrected settings record to {journal}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
