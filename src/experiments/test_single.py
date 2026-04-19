"""
test_single.py
--------------
End-to-end smoke test: pick ONE annotation, run the full Method A and
Method B pipeline against the live vLLM server, print a side-by-side
comparison with ground truth.

Usage:
    python src/experiments/test_single.py                    # default: first annotation
    python src/experiments/test_single.py --ann REBA_REBA_1_2.00s.json
    python src/experiments/test_single.py --method A         # only run Method A
    python src/experiments/test_single.py --method B         # only run Method B
    python src/experiments/test_single.py --prompt v1        # which prompt version

Preconditions:
    - vLLM server is running on localhost:8000 (see scripts/start_vllm_server.sh)
    - videos are at data/videos/
    - annotations are at data/annotations/
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.client import call_model, InferenceResult            # noqa: E402
from experiments.frame_extractor import extract_frame                 # noqa: E402
from reba_tables import compute_full_reba, RebaComputationError       # noqa: E402

ANN_DIR = PROJECT_ROOT / "data" / "annotations"
VID_DIR = PROJECT_ROOT / "data" / "videos"


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


def side_by_side(gt_scores: dict, pred_scores: dict):
    """Print a comparison table of scalar score fields."""
    fields = ["table_a", "score_a", "table_b", "score_b", "final_reba_score"]
    print(f"\n  {'field':22s} {'ground_truth':>14s} {'predicted':>12s}  match")
    print(f"  {'-'*22} {'-'*14:>14s} {'-'*12:>12s}  -----")
    for f in fields:
        g = gt_scores.get(f)
        p = pred_scores.get(f)
        mark = "✓" if g == p else "✗"
        print(f"  {f:22s} {str(g):>14s} {str(p):>12s}  {mark}")
    # action level
    g = (gt_scores.get("action_level") or {}).get("level")
    p = (pred_scores.get("action_level") or {}).get("level")
    mark = "✓" if g == p else "✗"
    print(f"  {'action_level.level':22s} {str(g):>14s} {str(p):>12s}  {mark}")


def part_comparison(gt_ann: dict, pred: dict):
    """Print per-part comparison of position_class and booleans."""
    print(f"\n  {'part':22s} {'field':25s} {'ground_truth':>30s} {'predicted':>30s}  match")
    print(f"  {'-'*22} {'-'*25} {'-'*30:>30s} {'-'*30:>30s}  -----")
    for group in ("group_a", "group_b", "context"):
        for part, part_gt in gt_ann.get(group, {}).items():
            part_pred = pred.get(group, {}).get(part, {})
            for field_name, gt_val in part_gt.items():
                if field_name == "sub_score":
                    continue
                pred_val = part_pred.get(field_name, "<missing>")
                mark = "✓" if gt_val == pred_val else "✗"
                print(f"  {part:22s} {field_name:25s} "
                      f"{str(gt_val):>30s} {str(pred_val):>30s}  {mark}")


def run_one(ann_file: Path, method: str, prompt_version: str):
    with open(ann_file, "r", encoding="utf-8") as f:
        ann = json.load(f)
    video_file = ann["meta_data"]["video_file"]
    ts = float(ann["meta_data"]["timestamp_sec"])
    video_path = VID_DIR / video_file

    build_prompt = load_prompt_builder(prompt_version)

    print(f"\n{'='*72}")
    print(f" Annotation : {ann_file.name}")
    print(f" Video      : {video_file}  (duration unknown, keyframe @ {ts}s)")
    print(f" Prompt     : {prompt_version}")
    print(f" Method     : {method}")
    print(f"{'='*72}")

    if method == "A":
        frame_path = extract_frame(video_path, ts)
        print(f"  keyframe jpeg: {frame_path}")
        prompt = build_prompt("A")
        result: InferenceResult = call_model(prompt, images=[frame_path])
    elif method == "B":
        prompt = build_prompt("B", timestamp=ts)
        result = call_model(prompt, video=video_path)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"\n  status          : {result.status}")
    print(f"  latency         : {result.latency_ms} ms")
    print(f"  prompt_tokens   : {result.prompt_tokens}")
    print(f"  completion_tok  : {result.completion_tokens}")

    if result.status == "http_error":
        print(f"  http_error      : {result.http_error}")
        return

    if result.status == "parse_failed":
        print(f"  parse_error     : {result.parse_error}")
        print(f"\n  --- raw response ---")
        print(result.raw_response)
        return

    # Both 'ok' and 'schema_invalid' reach here; schema_invalid still has
    # parsed_prediction populated, so per-field comparison is meaningful.
    pred = result.parsed_prediction
    if result.status == "schema_invalid":
        print(f"  parse_error     : {result.parse_error}")
        print("  (continuing with per-field comparison; skipping score computation)")

    part_comparison(ann, pred)

    if result.status == "ok":
        try:
            computed = compute_full_reba(pred, activity_score=0, strict=True)
        except RebaComputationError as e:
            print(f"\n  compute_full_reba failed: {e}")
            return
        side_by_side(ann["scores"], computed["scores"])
    else:
        print("\n  [scores side-by-side skipped — invalid enum(s) prevent lookup]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=None,
                    help="annotation filename; default = first one")
    ap.add_argument("--method", choices=["A", "B", "both"], default="both")
    ap.add_argument("--prompt", default="v1")
    args = ap.parse_args()

    if args.ann:
        ann_path = ANN_DIR / args.ann
    else:
        ann_path = sorted(ANN_DIR.glob("*.json"))[0]
    if not ann_path.exists():
        print(f"Annotation file not found: {ann_path}")
        sys.exit(1)

    methods = ["A", "B"] if args.method == "both" else [args.method]
    for m in methods:
        run_one(ann_path, m, args.prompt)


if __name__ == "__main__":
    main()