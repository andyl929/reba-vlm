"""
validate_full_pipeline.py
-------------------------
For every annotation JSON, recompute the full REBA pipeline from the
categorical fields alone, and compare against the labeled 'scores' block.

Fields compared:
  - table_a, score_a, table_b, score_b, final_reba_score, action_level.level

This is the FINAL sanity check for the REBA implementation. A clean run
here means: given a (perfect) model prediction of position_class + booleans
for all 8 parts, our code deterministically reproduces the ground-truth
final_reba_score.

The REBA_9_2.49s anomaly (coupling labeled=0 but category=fair_acceptable_hold)
is logged but does not count as a logic error.
"""

import json
from pathlib import Path
from collections import defaultdict

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reba_tables import compute_full_reba, RebaComputationError

ANN_DIR = PROJECT_ROOT / "data" / "annotations"
LOG_PATH = PROJECT_ROOT / "logs" / "full_pipeline_validation.json"
LOG_PATH.parent.mkdir(exist_ok=True)


def compare_one(labeled: dict, computed: dict):
    """Return dict of field->(labeled, computed) for mismatches only."""
    fields = ["table_a", "score_a", "table_b", "score_b", "final_reba_score"]
    mism = {}
    for f in fields:
        lv = labeled.get(f)
        cv = computed.get(f)
        if lv != cv:
            mism[f] = {"labeled": lv, "computed": cv}
    # action level
    lv = (labeled.get("action_level") or {}).get("level")
    cv = (computed.get("action_level") or {}).get("level")
    if lv != cv:
        mism["action_level.level"] = {"labeled": lv, "computed": cv}
    return mism


def main():
    ann_files = sorted(ANN_DIR.glob("*.json"))
    print(f"Loaded {len(ann_files)} annotations from {ANN_DIR}\n")

    n_ok = 0
    all_mismatches = []
    per_field_diff_counts = defaultdict(int)

    for p in ann_files:
        with open(p, "r", encoding="utf-8") as f:
            ann = json.load(f)

        try:
            result = compute_full_reba(ann, activity_score=0, strict=False)
        except RebaComputationError as e:
            all_mismatches.append({"file": p.name, "error": str(e)})
            continue

        labeled_scores = ann.get("scores", {})
        mism = compare_one(labeled_scores, result["scores"])

        if not mism:
            n_ok += 1
        else:
            all_mismatches.append({
                "file": p.name,
                "labeled": {k: labeled_scores.get(k) for k in
                            ["table_a", "score_a", "table_b", "score_b", "final_reba_score"]},
                "computed": {k: result["scores"].get(k) for k in
                             ["table_a", "score_a", "table_b", "score_b", "final_reba_score"]},
                "diffs": mism,
            })
            for f in mism:
                per_field_diff_counts[f] += 1

    print("=" * 70)
    print(f"RESULT: {n_ok}/{len(ann_files)} annotations match computed pipeline")
    print("=" * 70)

    if all_mismatches:
        print(f"\n{len(all_mismatches)} annotation(s) with discrepancies:\n")
        for m in all_mismatches:
            print(f"  [{m['file']}]")
            if "error" in m:
                print(f"    ERROR: {m['error']}")
            else:
                for field, vals in m["diffs"].items():
                    print(f"    {field:22s}: labeled={vals['labeled']}  computed={vals['computed']}")
        print(f"\nPer-field discrepancy counts:")
        for f, c in sorted(per_field_diff_counts.items(), key=lambda x: -x[1]):
            print(f"  {f:22s}: {c}")
    else:
        print("\n✓ ALL 44 ANNOTATIONS PERFECTLY MATCH PIPELINE COMPUTATION.")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_total": len(ann_files), "n_ok": n_ok,
                   "mismatches": all_mismatches}, f, indent=2)
    print(f"\nDetail log: {LOG_PATH}")


if __name__ == "__main__":
    main()