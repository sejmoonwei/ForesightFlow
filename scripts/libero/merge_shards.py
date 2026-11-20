#!/usr/bin/env python3
"""Merge sharded eval results into the standard ckpt_<step>.json format."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--ckpt-step", required=True)
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--expected", type=int, required=True,
                    help="Total number of shard JSONs to wait for before writing the merged ckpt file.")
    ap.add_argument("--ckpt-dir", default="")
    args = ap.parse_args()

    shards_dir = pathlib.Path(args.shards_dir)
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(shards_dir.glob("*.json"))
    if len(shard_files) < args.expected:
        print(f"[merge] only {len(shard_files)}/{args.expected} shards present; skip", file=sys.stderr)
        return 0

    per_suite_eps: dict[str, int] = collections.defaultdict(int)
    per_suite_succ: dict[str, int] = collections.defaultdict(int)
    elapsed_total = 0
    last_finished = ""
    logs_dirs: list[str] = []
    gpus_used: list[int] = []
    ports_used: list[int] = []
    num_trials_per_task = None

    for p in shard_files:
        with open(p) as f:
            d = json.load(f)
        suite = d["suite"]
        try:
            eps = int(d.get("total_episodes", 0) or 0)
            succ = int(d.get("total_successes", 0) or 0)
        except (TypeError, ValueError):
            eps, succ = 0, 0
        per_suite_eps[suite] += eps
        per_suite_succ[suite] += succ
        try:
            elapsed_total = max(elapsed_total, int(d.get("elapsed_seconds", 0) or 0))
        except (TypeError, ValueError):
            pass
        if d.get("finished_at", "") > last_finished:
            last_finished = d["finished_at"]
        logs_dirs.append(d.get("logs_dir", ""))
        try:
            gpus_used.append(int(d["gpu"]))
            ports_used.append(int(d["port"]))
        except (KeyError, TypeError, ValueError):
            pass
        if num_trials_per_task is None:
            num_trials_per_task = d.get("num_trials_per_task")

    suites_payload: dict[str, dict] = {}
    for suite, eps in per_suite_eps.items():
        succ = per_suite_succ[suite]
        rate = (succ / eps) if eps > 0 else float("nan")
        suites_payload[suite] = {
            "success_rate": rate,
            "total_episodes": eps,
            "total_successes": succ,
        }

    ckpt_step = args.ckpt_step
    ckpt_step_json: object
    if ckpt_step.isdigit():
        ckpt_step_json = int(ckpt_step)
    else:
        ckpt_step_json = ckpt_step

    out = {
        "ckpt_step": ckpt_step_json,
        "train_run": args.run_label,
        "ckpt_dir": args.ckpt_dir,
        "run_id": args.run_label,
        "gpu": ",".join(str(g) for g in gpus_used) if gpus_used else "?",
        "ports": ports_used,
        "num_trials_per_task": num_trials_per_task,
        "elapsed_seconds": elapsed_total,
        "finished_at": last_finished,
        "logs_dir": logs_dirs,
        "shards": [str(p) for p in shard_files],
        "suites": suites_payload,
    }

    out_file = results_dir / f"ckpt_{ckpt_step}.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[merge] wrote {out_file}  ({len(shard_files)} shards merged)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
