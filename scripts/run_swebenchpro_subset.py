import argparse
import json
from datetime import datetime
from pathlib import Path


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def compute_accuracy(jsonl_path: Path, expected_k: int) -> dict:
    total = 0
    passed = 0
    unevaluable = 0
    with jsonl_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                unevaluable += 1
                continue
            r = obj.get("reward")
            if r is None:
                unevaluable += 1
                continue
            try:
                if float(r) == 1.0:
                    passed += 1
            except Exception:
                unevaluable += 1
    denom_completed = total if total else 1
    denom_expected = expected_k if expected_k else 1
    evaluable_total = max(total - unevaluable, 0)
    denom_evaluable = evaluable_total if evaluable_total else 1
    return {
        "total": total,
        "passed": passed,
        "accuracy_completed": passed / denom_completed,
        "accuracy_expected": passed / denom_expected,
        "completion_rate": total / denom_expected,
        "evaluable_total": evaluable_total,
        "evaluable_rate": evaluable_total / denom_expected,
        "accuracy_evaluable": passed / denom_evaluable,
        "unevaluable": unevaluable,
        "jsonl_path": str(jsonl_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-path", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--traj-dir", default="./traj")
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--llm-name", default=None)
    ap.add_argument("--backend", default="docker")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-steps-absolute", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--use-fn-calling", action="store_true", default=False)
    ap.add_argument("--use-existing", action="store_true", default=False)
    ap.add_argument("--skip-existing", action="store_true", default=False)
    ap.add_argument("--max-reward-calc-time", type=int, default=180)
    ap.add_argument("--max-iterations", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--parallel", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--local-model-path", default=None)
    args = ap.parse_args()

    import pandas as pd
    from r2egym.agenthub.run.edit import runagent_multiple_parquet

    df = pd.read_parquet(args.parquet_path)
    if args.k is not None:
        k = int(args.k)
    else:
        k = int(len(df) * args.frac)
        if k < 1:
            k = 1
    exp_name = args.exp_name or f"swebenchpro_{int(args.frac*100)}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    llm_name = args.llm_name
    if not llm_name and args.local_model_path:
        llm_name = Path(args.local_model_path).name or "local"
    elif not llm_name:
        llm_name = "local"

    runagent_multiple_parquet(
        parquet_path=args.parquet_path,
        k=k,
        traj_dir=args.traj_dir,
        exp_name=exp_name,
        start_idx=args.start_idx,
        shuffle=args.shuffle,
        seed=args.seed,
        max_steps=args.max_steps,
        max_steps_absolute=args.max_steps_absolute,
        llm_name=llm_name,
        use_existing=args.use_existing,
        skip_existing=args.skip_existing,
        temperature=args.temperature,
        use_fn_calling=args.use_fn_calling,
        backend=args.backend,
        max_reward_calc_time=args.max_reward_calc_time,
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
        dry_run=args.dry_run,
        local_model_path=args.local_model_path,
    )

    jsonl_path = Path(args.traj_dir) / f"{exp_name}.jsonl"
    if jsonl_path.exists() and not args.dry_run:
        report = compute_accuracy(jsonl_path, expected_k=k)
        report["expected_k"] = k
        report["lines_in_file"] = _count_lines(jsonl_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "exp_name": exp_name,
                    "expected_k": k,
                    "jsonl_path": str(jsonl_path),
                    "dry_run": args.dry_run,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
