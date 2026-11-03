"""Print compact transcripts from Inspect logs for eyeballing runs.

    python analysis/transcripts.py logs/pilot                     # one-line score summary per sample
    python analysis/transcripts.py logs/pilot --full --only C2    # full transcripts for one condition
"""

import argparse

from inspect_ai.log import list_eval_logs, read_eval_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--full", action="store_true", help="print every tool call and result")
    ap.add_argument("--only", help="comma-separated conditions to include, e.g. C2,C4")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    for info in list_eval_logs(args.log_dir):
        log = read_eval_log(info)
        arm = log.eval.task_args.get("arm", "baseline")
        print(f"\n##### {log.eval.model}  arm={arm}  status={log.status}")
        for s in log.samples or []:
            if only and s.metadata["condition"] not in only:
                continue
            score = s.scores["ledger_scorer"]
            v = score.value
            flags = [k for k in ("false_success", "blind_retry", "unsafe_retry", "stale_trust", "no_report") if v[k]]
            print(
                f"{s.id:<16} truth={score.metadata['truth']:<19} claimed={score.metadata['claimed']:<19} "
                f"verified_all={v['verified_all_items']} {' '.join(flags)}"
            )
            if not args.full:
                continue
            for m in s.messages[2:]:
                if m.role == "assistant":
                    for tc in m.tool_calls or []:
                        print("     ->", tc.function, tc.arguments)
                    if m.text.strip():
                        print("     says:", m.text.strip()[:300])
                elif m.role == "tool":
                    print("     <-", (m.error.message if m.error else m.text)[:120])
                elif m.role == "user":
                    print("     [user]", m.text[:80])


if __name__ == "__main__":
    main()
