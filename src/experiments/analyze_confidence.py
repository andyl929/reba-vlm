"""
analyze_confidence.py
---------------------
v3-specific analysis: does the model's self-reported confidence correlate
with actual accuracy?

For each part's categorical prediction, split records by confidence level
(high vs medium vs low) and compute accuracy within each bin. If accuracy
is higher for "high" confidence than "medium", confidence is a useful
signal even if poorly calibrated in absolute terms.

Also computes:
  - Overall confidence distribution
  - Confidence difference between correct and incorrect predictions
    (calibration gap)
  - Pooled accuracy by confidence level (all parts combined)

Usage:
    python src/experiments/analyze_confidence.py --method A
    python src/experiments/analyze_confidence.py --method both
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

PART_SPECS = [
    ("group_a", "trunk", "position_class"),
    ("group_a", "neck", "position_class"),
    ("group_a", "legs", "position_class"),
    ("group_b", "upper_arms", "position_class"),
    ("group_b", "lower_arms", "position_class"),
    ("group_b", "wrists", "position_class"),
    ("context", "load_force", "category_class"),
    ("context", "coupling", "category_class"),
]


def load_jsonl(path):
    out = []
    if not path.exists():
        return out
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def get_nested(d, keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def analyze(records, method_label):
    print(f"\n{'=' * 72}")
    print(f" v3 / method {method_label}  (N = {len(records)})")
    print(f"{'=' * 72}")

    # =========================================================
    # 1. Per-part accuracy split by confidence
    # =========================================================
    print("\n--- Per-part: accuracy by self-reported confidence ---")
    print(f"  {'part':14s} {'conf':>7s} {'N':>4s} {'correct':>8s} {'acc':>7s}")
    print(f"  {'-'*14} {'-'*7:>7s} {'-'*4:>4s} {'-'*8:>8s} {'-'*7:>7s}")

    pooled = defaultdict(lambda: {"total": 0, "correct": 0})

    for group, part, enum_field in PART_SPECS:
        by_conf = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in records:
            pr = r["inference"].get("parsed_prediction")
            if not pr:
                continue
            gt_val = get_nested(r["ground_truth"], (group, part, enum_field))
            pr_val = get_nested(pr, (group, part, enum_field))
            if gt_val is None or pr_val is None:
                continue
            conf = (r["inference"]["v3_meta"]
                    ["per_part_confidence"].get(part, "unknown"))
            by_conf[conf]["total"] += 1
            if gt_val == pr_val:
                by_conf[conf]["correct"] += 1
            pooled[conf]["total"] += 1
            if gt_val == pr_val:
                pooled[conf]["correct"] += 1

        for conf_lvl in ("high", "medium", "low"):
            bucket = by_conf.get(conf_lvl)
            if not bucket or bucket["total"] == 0:
                continue
            acc = 100 * bucket["correct"] / bucket["total"]
            print(f"  {part:14s} {conf_lvl:>7s} {bucket['total']:>4d} "
                  f"{bucket['correct']:>8d} {acc:>6.1f}%")

    # =========================================================
    # 2. Pooled across all parts
    # =========================================================
    print("\n--- Pooled across all 8 parts ---")
    print(f"  {'confidence':>10s} {'N':>5s} {'correct':>9s} {'accuracy':>9s}")
    print(f"  {'-'*10:>10s} {'-'*5:>5s} {'-'*9:>9s} {'-'*9:>9s}")
    for conf_lvl in ("high", "medium", "low"):
        b = pooled.get(conf_lvl)
        if not b or b["total"] == 0:
            continue
        acc = 100 * b["correct"] / b["total"]
        print(f"  {conf_lvl:>10s} {b['total']:>5d} {b['correct']:>9d} {acc:>8.1f}%")

    # Calibration gap
    high_acc = (100 * pooled["high"]["correct"] / pooled["high"]["total"]
                if pooled["high"]["total"] else 0)
    med_acc = (100 * pooled["medium"]["correct"] / pooled["medium"]["total"]
               if pooled["medium"]["total"] else 0)
    if pooled["high"]["total"] and pooled["medium"]["total"]:
        gap = high_acc - med_acc
        direction = ("✓ useful signal" if gap > 5 else
                     "⚠ weak signal" if gap > 0 else
                     "✗ anti-correlated")
        print(f"\n  High-vs-medium accuracy gap: {gap:+.1f}pp  [{direction}]")

    # =========================================================
    # 3. Confidence distribution on correct vs incorrect predictions
    # =========================================================
    print("\n--- Confidence distribution split by correctness (pooled) ---")
    correct_confs = Counter()
    wrong_confs = Counter()
    for r in records:
        pr = r["inference"].get("parsed_prediction")
        if not pr:
            continue
        for group, part, enum_field in PART_SPECS:
            gt_val = get_nested(r["ground_truth"], (group, part, enum_field))
            pr_val = get_nested(pr, (group, part, enum_field))
            if gt_val is None or pr_val is None:
                continue
            conf = (r["inference"]["v3_meta"]
                    ["per_part_confidence"].get(part, "unknown"))
            if gt_val == pr_val:
                correct_confs[conf] += 1
            else:
                wrong_confs[conf] += 1

    total_correct = sum(correct_confs.values())
    total_wrong = sum(wrong_confs.values())
    print(f"  {'conf':>10s} {'correct':>10s} {'wrong':>10s} {'% correct':>11s}")
    for conf_lvl in ("high", "medium", "low"):
        c = correct_confs.get(conf_lvl, 0)
        w = wrong_confs.get(conf_lvl, 0)
        if c + w == 0:
            continue
        c_share = 100 * c / total_correct if total_correct else 0
        w_share = 100 * w / total_wrong if total_wrong else 0
        print(f"  {conf_lvl:>10s} {c:>5d} ({c_share:>4.1f}%) "
              f"{w:>4d} ({w_share:>4.1f}%) "
              f"{100*c/(c+w):>9.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    methods = ["A", "B"] if args.method == "both" else [args.method]
    for m in methods:
        path = RESULTS_DIR / f"v3_method_{m}.jsonl"
        recs = load_jsonl(path)
        if not recs:
            print(f"No records at {path}")
            continue
        analyze(recs, m)


if __name__ == "__main__":
    main()