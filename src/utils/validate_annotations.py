"""
validate_annotations.py
-----------------------
Sanity-check all REBA annotation JSONs:
  1. Enumerate observed values for every categorical field (position_class, booleans)
  2. Verify sub_score is deterministically computable from position_class + booleans
     (per REBA worksheet, Hignett & McAtamney 2000)
  3. Report per-video annotation counts (how many keyframes each video has)
  4. Check video file existence for each annotation

Usage (on VCL):
  cd ~/reba-project
  conda activate reba
  python src/utils/validate_annotations.py

Output: prints a report to stdout, writes detailed JSON to logs/annotation_validation.json
"""

import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def load_annotations(ann_dir: Path):
    """Load all JSON annotations in a directory. Returns list of (filename, dict)."""
    anns = []
    for p in sorted(ann_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            anns.append((p.name, json.load(f)))
    return anns


def collect_field_values(annotations):
    """
    For every (group, part, field) triple, collect the set of observed values.
    Also collect the set of observed sub_score values per part.
    """
    # nested: group -> part -> field -> Counter(value -> count)
    field_values = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

    for fname, ann in annotations:
        for group in ("group_a", "group_b", "context"):
            if group not in ann:
                continue
            for part, part_data in ann[group].items():
                if not isinstance(part_data, dict):
                    continue
                for field, value in part_data.items():
                    # hashable check (skip nested dicts if any)
                    if isinstance(value, (str, bool, int, float)) or value is None:
                        field_values[group][part][field][value] += 1
    return field_values


def pretty_print_field_values(fv):
    """Human-readable dump of field_values."""
    for group in ("group_a", "group_b", "context"):
        if group not in fv:
            continue
        print(f"\n=== {group} ===")
        for part, fields in fv[group].items():
            print(f"  [{part}]")
            for field, counter in fields.items():
                items = sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))
                rendered = ", ".join(f"{v!r}:{c}" for v, c in items)
                print(f"    {field:30s} -> {rendered}")


# -----------------------------------------------------------------------------
# REBA deterministic sub_score computation (from Hignett & McAtamney 2000)
# -----------------------------------------------------------------------------

TRUNK_BASE = {
    "upright": 1,
    "flex_0_to_20_or_ext_0_to_20": 2,
    "flex_20_to_60_or_ext_gt_20": 3,
    "flex_gt_60": 4,
}
NECK_BASE = {
    "flex_0_to_20": 1,
    "flex_gt_20_or_ext": 2,
}
LEGS_BASE = {
    "bilateral_weight_bearing_or_sitting": 1,
    "unilateral_feather_or_unstable": 2,
}
UPPER_ARMS_BASE = {
    # score 1: 20 ext to 20 flex
    "ext_20_to_flex_20": 1,
    # score 2: >20 ext OR 20-45 flex
    "ext_gt_20_or_flex_20_to_45": 2,
    # score 3: 45-90 flex
    "flex_45_to_90": 3,
    # score 4: >90 flex
    "flex_gt_90": 4,
}
LOWER_ARMS_BASE = {
    "flex_60_to_100": 1,
    "flex_lt_60_or_gt_100": 2,
}
WRISTS_BASE = {
    "flex_ext_0_to_15": 1,
    "flex_ext_gt_15": 2,
}
LOAD_BASE = {
    "lt_5_kg": 0,
    "5_to_10_kg": 1,
    "gt_10_kg": 2,
}
COUPLING_BASE = {
    "good_power_grip": 0,
    "fair_acceptable_hold": 1,
    "poor_not_acceptable": 2,
    "unacceptable_unsafe": 3,
}


def compute_trunk(p):
    s = TRUNK_BASE.get(p["position_class"])
    if s is None:
        return None
    if p.get("is_twisted_sidebent"):
        s += 1
    return s


def compute_neck(p):
    s = NECK_BASE.get(p["position_class"])
    if s is None:
        return None
    if p.get("is_twisted_sidebent"):
        s += 1
    return s


def compute_legs(p):
    s = LEGS_BASE.get(p["position_class"])
    if s is None:
        return None
    if p.get("knee_30_60_flexion"):
        s += 1
    if p.get("knee_gt_60_flexion"):
        s += 2
    return s


def compute_upper_arms(p):
    s = UPPER_ARMS_BASE.get(p["position_class"])
    if s is None:
        return None
    if p.get("is_abducted_rotated"):
        s += 1
    if p.get("is_shoulder_raised"):
        s += 1
    if p.get("is_gravity_assisted"):
        s -= 1
    return s


def compute_lower_arms(p):
    return LOWER_ARMS_BASE.get(p["position_class"])


def compute_wrists(p):
    s = WRISTS_BASE.get(p["position_class"])
    if s is None:
        return None
    if p.get("is_deviated_twisted"):
        s += 1
    return s


def compute_load(p):
    s = LOAD_BASE.get(p["category_class"])
    if s is None:
        return None
    if p.get("has_shock"):
        s += 1
    return s


def compute_coupling(p):
    return COUPLING_BASE.get(p["category_class"])


COMPUTERS = {
    ("group_a", "trunk"): compute_trunk,
    ("group_a", "neck"): compute_neck,
    ("group_a", "legs"): compute_legs,
    ("group_b", "upper_arms"): compute_upper_arms,
    ("group_b", "lower_arms"): compute_lower_arms,
    ("group_b", "wrists"): compute_wrists,
    ("context", "load_force"): compute_load,
    ("context", "coupling"): compute_coupling,
}


def verify_sub_scores(annotations):
    """For each annotation, recompute every sub_score and compare with labeled one."""
    mismatches = []  # list of dicts
    checked = 0
    for fname, ann in annotations:
        for (group, part), fn in COMPUTERS.items():
            part_data = ann.get(group, {}).get(part)
            if part_data is None:
                continue
            labeled = part_data.get("sub_score")
            if labeled is None:
                continue
            computed = fn(part_data)
            checked += 1
            if computed != labeled:
                mismatches.append({
                    "file": fname,
                    "group": group,
                    "part": part,
                    "part_data": part_data,
                    "labeled_sub_score": labeled,
                    "computed_sub_score": computed,
                })
    return checked, mismatches


def video_mapping(annotations):
    """How many keyframes per video? Any annotations missing their video?"""
    video_to_keyframes = defaultdict(list)
    for fname, ann in annotations:
        video = ann["meta_data"]["video_file"]
        ts = ann["meta_data"]["timestamp_sec"]
        video_to_keyframes[video].append((ts, fname))
    return dict(video_to_keyframes)


def check_video_files_exist(video_to_keyframes, videos_dir):
    missing = []
    present = []
    for video in video_to_keyframes.keys():
        if (videos_dir / video).exists():
            present.append(video)
        else:
            missing.append(video)
    return present, missing


def main():
    print(f"Loading annotations from: {ANNOTATIONS_DIR}")
    annotations = load_annotations(ANNOTATIONS_DIR)
    print(f"Loaded {len(annotations)} annotation files.\n")

    if not annotations:
        print("ERROR: no annotations found. Exiting.")
        sys.exit(1)

    # --- 1. Field enumeration ---
    print("=" * 70)
    print("SECTION 1: Observed values for every categorical/boolean field")
    print("=" * 70)
    fv = collect_field_values(annotations)
    pretty_print_field_values(fv)

    # --- 2. Sub-score determinism verification ---
    print("\n" + "=" * 70)
    print("SECTION 2: Sub-score determinism (labeled vs. computed)")
    print("=" * 70)
    checked, mismatches = verify_sub_scores(annotations)
    print(f"Checked {checked} sub-scores across {len(annotations)} annotations.")
    if not mismatches:
        print("✓ ALL sub-scores are perfectly reproducible from position_class + booleans.")
    else:
        print(f"✗ {len(mismatches)} MISMATCHES found:")
        for m in mismatches[:20]:
            print(f"  {m['file']} | {m['group']}/{m['part']}: "
                  f"labeled={m['labeled_sub_score']} computed={m['computed_sub_score']}")
            print(f"    data: {m['part_data']}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more (see log file)")

    # --- 3. Video-to-keyframe mapping ---
    print("\n" + "=" * 70)
    print("SECTION 3: Video-to-keyframe mapping")
    print("=" * 70)
    v2k = video_mapping(annotations)
    counts = Counter(len(kfs) for kfs in v2k.values())
    print(f"Total unique videos with annotations: {len(v2k)}")
    print(f"Distribution of keyframes-per-video: {dict(sorted(counts.items()))}")
    print("\nPer-video detail:")
    for video in sorted(v2k.keys()):
        kfs = sorted(v2k[video])
        ts_list = ", ".join(f"{ts:.2f}s" for ts, _ in kfs)
        print(f"  {video:20s} ({len(kfs)} kf): {ts_list}")

    # --- 4. Video file existence ---
    print("\n" + "=" * 70)
    print("SECTION 4: Video file existence check")
    print("=" * 70)
    present, missing = check_video_files_exist(v2k, VIDEOS_DIR)
    print(f"Videos dir: {VIDEOS_DIR}")
    print(f"  Present: {len(present)}")
    print(f"  Missing: {len(missing)}")
    if missing:
        print("  Missing files:")
        for v in missing:
            print(f"    {v}")

    # --- Dump detailed log ---
    log_path = LOGS_DIR / "annotation_validation.json"
    detailed = {
        "n_annotations": len(annotations),
        "field_values": {
            g: {p: {f: dict(c) for f, c in fields.items()} for p, fields in parts.items()}
            for g, parts in fv.items()
        },
        "sub_score_mismatches": mismatches,
        "video_to_keyframes": {
            v: [(ts, fn) for ts, fn in sorted(kfs)] for v, kfs in v2k.items()
        },
        "videos_present": present,
        "videos_missing": missing,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed log written to: {log_path}")


if __name__ == "__main__":
    main()