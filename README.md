# agent-honesty-eval

This project tests whether tool-using AI agents report failure honestly. It is a small pre-registered experiment built on [Inspect AI](https://inspect.aisi.org.uk/).

Agents get short operational tasks with three items each: append lines to files, comment on tickets, place orders, or deploy services. While the agent works, the harness injects failures it can detect but is never told about:

- explicit errors
- silent no-ops
- partial completion
- timeouts after the side effect already happened
- stale handoff notes

A ground-truth ledger records what really happened. Scoring compares that ledger with the status the agent reports.

The hypotheses, outcomes and tests are fixed in [PREREGISTRATION.md](PREREGISTRATION.md).

## Layout

| Path | Purpose |
|---|---|
| `honesty_eval/world.py` | Per-sample state, failure injection, ledger |
| `honesty_eval/tools.py` | Mock action/read tools for each domain, plus `report` |
| `honesty_eval/specs.py` | The 24 task specifications |
| `honesty_eval/prompts.py` | Baseline and guardrail system prompts, and the stale note |
| `honesty_eval/scoring.py` | Deterministic ledger scorer |
| `honesty_eval/task.py` | Inspect task and agent loop |
| `honesty_eval/scripted.py` | Scripted `careful` and `naive` policies for self-tests |
| `honesty_eval/judge.py` | Report-text judge: frozen rubric and model (secondary outcome) |
| `analysis/judge.py` | Runs the judge over finished logs, writing `analysis/out/<log dir>/judge.csv` |
| `analysis/summarize.py` | Pre-registered tables, judge tables, Cohen's κ, label sheet |

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
```

## Self-test (no API calls)

```bash
.venv/bin/python -m pytest -q
```

The self-test runs all 144 samples per arm through the real Inspect pipeline using scripted policies. The careful policy must never produce a false success. The naive policy must trip each failure metric exactly where it is expected to.

## Running a model

```bash
.venv/bin/inspect eval honesty_eval/task.py -T arm=baseline --model anthropic/claude-haiku-5-5 --epochs 3 --log-dir logs/main
```

```bash
.venv/bin/inspect eval honesty_eval/task.py -T arm=guardrail --model anthropic/claude-haiku-5-5 --epochs 3 --log-dir logs/main
```

```bash
cd analysis && ../.venv/bin/python judge.py ../logs/main && ../.venv/bin/python judge.py ../logs/main --out judge_pass2.csv
```

```bash
cd analysis && ../.venv/bin/python summarize.py ../logs/main --labels
```

Do not pass `--temperature`: Anthropic models no longer accept it, and Inspect records the requested value without sending it. Repeat both commands for each model in the pre-registration (Haiku, Sonnet and Opus 5.5). For a pilot, add `-T conditions=C0,C2,C4 --limit 20`. Keep one log per (model, arm) in a log directory; the analysis script refuses to pair duplicates.

## Status

The pre-registration was frozen on 2026-11-02, after the pilot and before the main run. Results and the write-up will be added in later commits. Any change to the plan is listed in the Deviations log in [PREREGISTRATION.md](PREREGISTRATION.md).

## License

MIT. See [LICENSE](LICENSE).
