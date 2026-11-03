"""Label every report summary in a log directory with the frozen judge.

    python analysis/judge.py logs/main                   # writes analysis/out/main/judge.csv
    python analysis/judge.py logs/main --out judge_pass2.csv   # second pass, for self-agreement
    python analysis/judge.py logs/main --model mockllm/model   # dry run, no API calls

Outputs go to analysis/out/<log dir name>/. Rows already in the output are skipped, so an interrupted run can be resumed.
"""

import argparse
import asyncio
import csv

from summarize import load_rows, out_dir

from honesty_eval.judge import JUDGE_MODEL, judge_model, judge_summary

FIELDS = ["model", "arm", "sample_id", "epoch", "judge_model", "judge_label", "judge_raw"]


def row_key(r: dict) -> tuple[str, str, str, str]:
    return (r["model"], r["arm"], str(r["sample_id"]), str(r["epoch"]))


async def run(log_dir: str, model_name: str, concurrency: int, out: str) -> None:
    path = out_dir(log_dir) / out
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with open(path, newline="") as f:
            done = {row_key(r) for r in csv.DictReader(f)}
    todo = [r for r in load_rows(log_dir) if row_key(r) not in done]
    print(f"{len(done)} already judged, {len(todo)} to go with {model_name}")

    model = judge_model(model_name)
    sem = asyncio.Semaphore(concurrency)

    async def one(r: dict) -> dict:
        async with sem:
            label, raw = await judge_summary(r["request"], r["summary"], model)
        return {**{k: r[k] for k in FIELDS[:4]}, "judge_model": model_name, "judge_label": label, "judge_raw": raw}

    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for fut in asyncio.as_completed([one(r) for r in todo]):
            w.writerow(await fut)
            f.flush()
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="judge.csv", help="file name under analysis/out/<log dir name>")
    args = ap.parse_args()
    asyncio.run(run(args.log_dir, args.model, args.concurrency, args.out))


if __name__ == "__main__":
    main()
