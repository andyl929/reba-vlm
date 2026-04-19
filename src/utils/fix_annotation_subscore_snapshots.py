"""
fix_annotation_subscore_snapshots.py
------------------------------------
Repair the rare case where an annotation JSON's stored `sub_score` is an
outdated Streamlit-rerun snapshot that disagrees with its own
position_class/category_class + boolean adjustments.

Strategy:
  - For every annotation, recompute every part's sub_score deterministically
    from reba_tables.py.
  - If the stored value differs, record it as a repair.
  - Also recompute the full 'scores' block; flag (but don't auto-fix) any
    discrepancy there, because the current 44/44 pipeline match means these
    should already be correct — any discrepancy here is a NEW warning worth
    human inspection.
  - In --dry-run mode (default): print everything, write nothing.
  - With --apply: make a timestamped backup (.bak-YYYYMMDD_HHMMSS) then
    overwrite the JSON with the fully recomputed version.

Only fields that already exist in the JSON are modified; no schema drift.

Usage:
    python src/utils/fix_annotation_subscore_snapshots.py           # dry run
    python src/utils/fix_annotation_subscore_snapshots.py --apply   # actually fix
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reba_tables import (  # noqa: E402
    compute_trunk, compute_neck, compute_legs,
    compute_upper_arms, compute_lower_arms, compute_wrists,
    compute_load, compute_coupling, compute_full_reba,
)

ANN_DIR = PROJECT_ROOT / "data" / "annotations"

# (group, part, computer_fn)
PART_COMPUTERS = [
    ("group_a", "trunk", compute_trunk),
    ("group_a", "neck", compute_neck),
    ("group_a", "legs", compute_legs),
    ("group_b", "upper_arms", compute_upper_arms),
    ("group_b", "lower_arms", compute_lower_arms),
    ("group_b", "wrists", compute_wrists),
    ("context", "load_force", compute_load),
    ("context", "coupling", compute_coupling),
]


def analyze(ann: dict):
    """
    Returns (sub_score_repairs, score_block_warnings) for one annotation.
      sub_score_repairs: list of (group, part, old, new)
      score_block_warnings: list of (field, old, new)  [not auto-fixed]
    """
    repairs = []
    for group, part, fn in PART_COMPUTERS:
        part_data = ann.get(group, {}).get(part)
        if not part_data:
            continue
        old = part_data.get("sub_score")
        new = fn(part_data)
        if old != new:
            repairs.append((group, part, old, new))

    # cross-check the top-level scores block
    warnings = []
    try:
        recomputed = compute_full_reba(ann, activity_score=0, strict=True)
    except Exception as e:
        warnings.append(("<compute_full_reba-error>", None, str(e)))
        return repairs, warnings

    labeled_scores = ann.get("scores", {}) or {}
    for f in ("table_a", "score_a", "table_b", "score_b", "final_reba_score"):
        old = labeled_scores.get(f)
        new = recomputed["scores"].get(f)
        if old != new:
            warnings.append((f, old, new))
    labeled_lvl = (labeled_scores.get("action_level") or {}).get("level")
    new_lvl = (recomputed["scores"].get("action_level") or {}).get("level")
    if labeled_lvl != new_lvl:
        warnings.append(("action_level.level", labeled_lvl, new_lvl))

    return repairs, warnings


def apply_repairs(ann: dict, repairs):
    for group, part, _old, new in repairs:
        ann[group][part]["sub_score"] = new
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually modify files (make .bak backups). Default is dry-run.")
    args = ap.parse_args()

    ann_files = sorted(ANN_DIR.glob("*.json"))
    print(f"Scanning {len(ann_files)} annotations in {ANN_DIR}")
    print(f"Mode: {'APPLY (will modify files)' if args.apply else 'DRY RUN (no files modified)'}\n")

    total_files_with_subscore_drift = 0
    total_files_with_score_block_drift = 0

    for p in ann_files:
        with open(p, "r", encoding="utf-8") as f:
            ann = json.load(f)

        repairs, warnings = analyze(ann)

        if repairs or warnings:
            print(f"[{p.name}]")
            for group, part, old, new in repairs:
                print(f"  FIX  {group}/{part}.sub_score: {old} -> {new}")
            for field, old, new in warnings:
                print(f"  WARN score.{field}: labeled={old} computed={new}   "
                      f"(NOT auto-fixed; inspect manually)")

        if repairs:
            total_files_with_subscore_drift += 1
            if args.apply:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = p.with_suffix(p.suffix + f".bak-{stamp}")
                shutil.copy2(p, backup)
                fixed = apply_repairs(ann, repairs)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(fixed, f, ensure_ascii=False, indent=2)
                print(f"  ✓ applied, backup at {backup.name}")

        if warnings:
            total_files_with_score_block_drift += 1

    print("\n" + "=" * 70)
    print(f"Files with sub_score snapshot drift : {total_files_with_subscore_drift}")
    print(f"Files with 'scores' block drift     : {total_files_with_score_block_drift}  "
          f"(these are NOT auto-fixed)")
    if not args.apply and total_files_with_subscore_drift > 0:
        print("\nRe-run with --apply to actually modify files.")
    if total_files_with_score_block_drift > 0:
        print("\nWARNING: 'scores' block drift found. These are not mere stale snapshots;\n"
              "         the ground-truth computation itself is inconsistent. INVESTIGATE\n"
              "         before running experiments.")


if __name__ == "__main__":
    main()