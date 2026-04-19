"""
run_experiment.py
-----------------
Batch-run REBA inference for all annotations, writing one JSONL per
(prompt_version, method) condition. Supports resume: re-invoking picks up
where a previous crash left off.

Usage:
    python src/experiments/run_experiment.py --prompt v1 --method A
    python src/experiments/run_experiment.py --prompt v1 --method B
    python src/experiments/run_experiment.py --prompt v1 --method both
    python src/experiments/run_experiment.py --prompt v1 --method A --limit 3     # smoke test
    python src/experiments/run_experiment.py --prompt v1 --method A --force       # re-run all

Output format (per JSONL line):
    {
      "annotation_file": str,
      "video_file": str,
      "timestamp_sec": float,
      "prompt_version": str,
      "method": "A" | "B",
      "ground_truth": {...},           # full annotation JSON
      "inference": {
        "status": "ok" | "schema_invalid" | "parse_failed" | "http_error",
        "latency_ms": int,
        "prompt_tokens": int|null,
        "completion_tokens": int|null,
        "raw_response": str,
        "parsed_prediction": dict|null,
        "parse_error": str|null
      },
      "computed_scores": {...}|null    # filled only if status == "ok"
    }
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.client import call_model                             # noqa: E402
from experiments.frame_extractor import extract_frame                 # noqa: E402
from reba_tables import compute_full_reba, RebaComputationError       # noqa: E402

ANN_DIR = PROJECT_ROOT / "data" / "annotations"
VID_DIR = PROJECT_ROOT / "data" / "videos"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_prompt_builder(version: str):
    if version == "v1":
        from prompts.v1_baseline import build_prompt
        return build_prompt
    if version == "v2":
        from prompts.v2_detailed import build_prompt
        return build_prompt
    if version == "v2_1":
        from prompts.v2_1_detailed import build_prompt
        return build_prompt
    raise ValueError(f"Prompt version not implemented yet: {version}")


def results_path(prompt_version: str, method: str) -> Path:
    return RESULTS_DIR / f"{prompt_version}_method_{method}.jsonl"


def already_done(path: Path) -> Set[str]:
    """Return the set of annotation_file values that already appear in path."""
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["annotation_file"])
            except (json.JSONDecodeError, KeyError):
                continue  # skip malformed lines; the rerun will redo them
    return done


def run_condition(prompt_version: str, method: str, limit: int = 0,
                  force: bool = False) -> dict:
    """Run one (prompt, method) condition over all annotations. Returns stats."""
    is_v3 = prompt_version == "v3"
    if not is_v3:
        build_prompt = load_prompt_builder(prompt_version)
    else:
        # v3 uses the orchestrator, not a simple prompt builder
        from experiments.v3_orchestrator import run_v3_single, to_dict as v3_to_dict
    out_path = results_path(prompt_version, method)
    done = set() if force else already_done(out_path)

    if force and out_path.exists():
        out_path.unlink()
        done = set()

    annotation_files = sorted(ANN_DIR.glob("*.json"))
    todo = [p for p in annotation_files if p.name not in done]
    if limit > 0:
        todo = todo[:limit]

    total = len(todo)
    print(f"\n[{prompt_version}/method-{method}]")
    print(f"  output      : {out_path}")
    print(f"  already done: {len(done)}")
    print(f"  to run      : {total}")
    if total == 0:
        print("  (nothing to do)")
        return {"condition": f"{prompt_version}/{method}", "skipped": True}

    stats = {"ok": 0, "schema_invalid": 0, "parse_failed": 0, "http_error": 0}
    t_start = time.perf_counter()

    with open(out_path, "a", encoding="utf-8") as fout:
        for idx, ann_path in enumerate(todo, 1):
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            video_file = ann["meta_data"]["video_file"]
            ts = float(ann["meta_data"]["timestamp_sec"])
            video_path = VID_DIR / video_file

            # build media inputs & call model
            if is_v3:
                if method == "A":
                    frame_path = extract_frame(video_path, ts)
                    v3_result = run_v3_single(
                        image_paths=[frame_path], method="A", timestamp=ts,
                    )
                elif method == "B":
                    v3_result = run_v3_single(
                        video_path=video_path, method="B", timestamp=ts,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                # Wrap v3 result in a dict compatible with downstream
                inference_dict = v3_to_dict(v3_result)
                # Build a shim that looks like InferenceResult for the rest
                class _Shim:
                    status = inference_dict["status"]
                    latency_ms = inference_dict["latency_ms"]
                    parsed_prediction = inference_dict["parsed_prediction"]
                    parse_error = inference_dict["parse_error"]
                result = _Shim()
                result.status = inference_dict["status"]
                result.latency_ms = inference_dict["latency_ms"]
                result.parsed_prediction = inference_dict["parsed_prediction"]
                result.parse_error = inference_dict["parse_error"]
            else:
                if method == "A":
                    frame_path = extract_frame(video_path, ts)
                    prompt = build_prompt("A")
                    result = call_model(prompt, images=[frame_path])
                elif method == "B":
                    prompt = build_prompt("B", timestamp=ts)
                    result = call_model(prompt, video=video_path)
                else:
                    raise ValueError(f"Unknown method: {method}")

            stats[result.status] = stats.get(result.status, 0) + 1

            # compute scores if parse was valid
            computed_scores = None
            if result.status == "ok":
                try:
                    computed = compute_full_reba(result.parsed_prediction,
                                                 activity_score=0, strict=True)
                    computed_scores = computed["scores"]
                except RebaComputationError as e:
                    # shouldn't happen given validate_prediction passed, but be safe
                    stats["ok"] -= 1
                    stats["schema_invalid"] = stats.get("schema_invalid", 0) + 1
                    result.status = "schema_invalid"
                    result.parse_error = f"post-validation compute error: {e}"

            rec = {
                "annotation_file": ann_path.name,
                "video_file": video_file,
                "timestamp_sec": ts,
                "prompt_version": prompt_version,
                "method": method,
                "ground_truth": ann,
                "inference": (inference_dict if is_v3 else result.to_dict()),
                "computed_scores": computed_scores,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            # progress line
            elapsed = time.perf_counter() - t_start
            per_item = elapsed / idx
            eta = per_item * (total - idx)
            ok_so_far = stats["ok"]
            valid_pred = stats["ok"] + stats["schema_invalid"]
            print(f"  [{idx:3d}/{total}] {ann_path.name:32s} "
                  f"{result.status:14s} {result.latency_ms:>6d}ms  "
                  f"ok={ok_so_far}/{idx}  "
                  f"parsed={valid_pred}/{idx}  "
                  f"ETA {eta/60:.1f}m", flush=True)

    dt = time.perf_counter() - t_start
    print(f"\n  done in {dt/60:.1f} min. stats: {stats}")
    return {
        "condition": f"{prompt_version}/{method}",
        "total_run_this_session": total,
        "total_in_file": len(done) + total,
        "stats": stats,
        "elapsed_sec": int(dt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--method", choices=["A", "B", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N annotations (0 = all)")
    ap.add_argument("--force", action="store_true",
                    help="delete existing results file(s) and start fresh")
    args = ap.parse_args()

    methods = ["A", "B"] if args.method == "both" else [args.method]
    overall = []
    for m in methods:
        stats = run_condition(args.prompt, m, limit=args.limit, force=args.force)
        overall.append(stats)

    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    for s in overall:
        print(f"  {s['condition']}: {s.get('stats', s)}")


if __name__ == "__main__":
    main()