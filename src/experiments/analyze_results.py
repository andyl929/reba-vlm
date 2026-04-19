"""
analyze_results.py
------------------
Comprehensive analysis of a single (prompt_version, method) result file,
or cross-method comparison.

Produces:
  1. Parse/schema outcome breakdown
  2. Per-field accuracy (for ALL results with parsed_prediction, whether
     schema_invalid or ok — a bad enum on one field doesn't invalidate the
     other 16 predictions)
  3. Confusion matrices for every categorical field
  4. Boolean-field error directionality (false-negative vs false-positive)
  5. Score-level metrics on the strict 'ok' subset (MAE, exact-match,
     correlation for final_reba_score, action_level confusion)
  6. Per-annotation difficulty ranking (most-wrong samples)
  7. Method A vs Method B side-by-side summary

Usage:
    python src/experiments/analyze_results.py --prompt v1
    python src/experiments/analyze_results.py --prompt v1 --method A     # just one
    python src/experiments/analyze_results.py --prompt v1 --out-dir analysis/
"""

import argparse
import json
import sys
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reba_tables import compute_full_reba, RebaComputationError, ENUMS  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUT_DIR = PROJECT_ROOT / "analysis"


# Field registry, in fixed display order.
CATEGORICAL_FIELDS = [
    ("group_a", "trunk",      "position_class"),
    ("group_a", "neck",       "position_class"),
    ("group_a", "legs",       "position_class"),
    ("group_b", "upper_arms", "position_class"),
    ("group_b", "lower_arms", "position_class"),
    ("group_b", "wrists",     "position_class"),
    ("context", "load_force", "category_class"),
    ("context", "coupling",   "category_class"),
]

BOOLEAN_FIELDS = [
    ("group_a", "trunk",      "is_twisted_sidebent"),
    ("group_a", "neck",       "is_twisted_sidebent"),
    ("group_a", "legs",       "knee_30_60_flexion"),
    ("group_a", "legs",       "knee_gt_60_flexion"),
    ("group_b", "upper_arms", "is_abducted_rotated"),
    ("group_b", "upper_arms", "is_shoulder_raised"),
    ("group_b", "upper_arms", "is_gravity_assisted"),
    ("group_b", "wrists",     "is_deviated_twisted"),
    ("context", "load_force", "has_shock"),
]


def load_jsonl(path: Path) -> List[dict]:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_nested(d: dict, path: Tuple[str, ...], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# =============================================================================
# Section 1 — outcome breakdown
# =============================================================================

def section_outcomes(records: List[dict], label: str) -> Dict[str, Any]:
    print(f"\n{'=' * 72}")
    print(f" [{label}] Parse/Schema Outcomes  (N = {len(records)})")
    print(f"{'=' * 72}")
    c = Counter(r["inference"]["status"] for r in records)
    for status in ("ok", "schema_invalid", "parse_failed", "http_error"):
        n = c.get(status, 0)
        pct = 100 * n / max(1, len(records))
        print(f"  {status:18s}: {n:3d}  ({pct:5.1f}%)")

    # what % of schema_invalid is caused by each field?
    inv = [r for r in records if r["inference"]["status"] == "schema_invalid"]
    if inv:
        reason_counter = Counter()
        for r in inv:
            err = r["inference"].get("parse_error") or ""
            # extract the field mentioned; e.g. "group_b.upper_arms.position_class=..."
            # naive first-field extraction
            field = err.split(" ")[0] if err else "<unknown>"
            reason_counter[field] += 1
        print(f"\n  Schema-invalid root-cause fields (first error in each):")
        for field, n in reason_counter.most_common():
            print(f"    {field:50s}: {n}")

    return dict(c)


# =============================================================================
# Section 2 — per-field accuracy across all parsed predictions
# =============================================================================

def section_field_accuracy(records: List[dict], label: str):
    """For each of 17 fields, count matches / total where both sides exist."""
    print(f"\n{'=' * 72}")
    print(f" [{label}] Per-field accuracy  (uses ALL records with parsed_prediction)")
    print(f"{'=' * 72}")

    rows = []
    parsed = [r for r in records if r["inference"].get("parsed_prediction")]

    def field_stats(path: Tuple[str, ...]):
        n, correct = 0, 0
        for r in parsed:
            gt = get_nested(r["ground_truth"], path)
            pred = get_nested(r["inference"]["parsed_prediction"], path)
            if gt is None or pred is None:
                continue
            n += 1
            if gt == pred:
                correct += 1
        return correct, n

    print(f"  {'field':40s} {'correct':>8s} / {'total':<6s} {'acc':>8s}")
    print(f"  {'-' * 40} {'-' * 8:>8s} / {'-' * 6:<6s} {'-' * 8:>8s}")

    cat_rows = []
    for g, p, f in CATEGORICAL_FIELDS:
        c, n = field_stats((g, p, f))
        name = f"{p}.{f}"
        acc = 100 * c / n if n else 0.0
        cat_rows.append((name, c, n, acc))
        print(f"  {name:40s} {c:>8d} / {n:<6d} {acc:>7.1f}%")

    print(f"  {'':40s}")
    bool_rows = []
    for g, p, f in BOOLEAN_FIELDS:
        c, n = field_stats((g, p, f))
        name = f"{p}.{f}"
        acc = 100 * c / n if n else 0.0
        bool_rows.append((name, c, n, acc))
        print(f"  {name:40s} {c:>8d} / {n:<6d} {acc:>7.1f}%")

    cat_total_c = sum(r[1] for r in cat_rows)
    cat_total_n = sum(r[2] for r in cat_rows)
    bool_total_c = sum(r[1] for r in bool_rows)
    bool_total_n = sum(r[2] for r in bool_rows)
    print(f"\n  {'[categorical mean]':40s} {cat_total_c:>8d} / {cat_total_n:<6d} "
          f"{100*cat_total_c/max(1,cat_total_n):>7.1f}%")
    print(f"  {'[boolean mean]':40s} {bool_total_c:>8d} / {bool_total_n:<6d} "
          f"{100*bool_total_c/max(1,bool_total_n):>7.1f}%")
    total_c = cat_total_c + bool_total_c
    total_n = cat_total_n + bool_total_n
    print(f"  {'[all fields mean]':40s} {total_c:>8d} / {total_n:<6d} "
          f"{100*total_c/max(1,total_n):>7.1f}%")

    return {"categorical": cat_rows, "boolean": bool_rows}


# =============================================================================
# Section 3 — confusion matrices for categorical fields
# =============================================================================

def section_confusion(records: List[dict], label: str):
    print(f"\n{'=' * 72}")
    print(f" [{label}] Confusion matrices  (rows = ground_truth, cols = predicted)")
    print(f"{'=' * 72}")

    parsed = [r for r in records if r["inference"].get("parsed_prediction")]

    for (g, p, f) in CATEGORICAL_FIELDS:
        enum_key = f"{p}.{f}"
        allowed = ENUMS[enum_key]
        print(f"\n  --- {p}.{f} ---")

        # Build GT -> Counter(pred)
        cm = defaultdict(Counter)
        gt_values_seen = set()
        for r in parsed:
            gt = get_nested(r["ground_truth"], (g, p, f))
            pr = get_nested(r["inference"]["parsed_prediction"], (g, p, f))
            if gt is None or pr is None:
                continue
            cm[gt][pr] += 1
            gt_values_seen.add(gt)

        # All prediction values observed (may include invalid enum values)
        all_preds = sorted({pred for row in cm.values() for pred in row})

        # print header: abbreviated prediction labels
        def abbr(s: str, n=18):
            return s if len(s) <= n else s[:n - 1] + "…"

        col_headers = [abbr(p_, 18) for p_ in all_preds]
        col_widths = [max(4, len(h)) for h in col_headers]
        header = "  " + " " * 42 + "  ".join(f"{h:>{w}}" for h, w in zip(col_headers, col_widths))
        print(header)
        for gt_val in allowed:
            if gt_val not in gt_values_seen:
                continue
            row = cm[gt_val]
            cells = []
            for pred_val, w in zip(all_preds, col_widths):
                c = row.get(pred_val, 0)
                cells.append(f"{c:>{w}d}" if c else " " * w)
            total = sum(row.values())
            correct = row.get(gt_val, 0)
            line = f"  {abbr(gt_val, 40):<40s}  " + "  ".join(cells) + f"   ({correct}/{total})"
            print(line)


# =============================================================================
# Section 4 — boolean error directionality
# =============================================================================

def section_bool_directionality(records: List[dict], label: str):
    print(f"\n{'=' * 72}")
    print(f" [{label}] Boolean error analysis  (FN = missed true, FP = hallucinated)")
    print(f"{'=' * 72}")

    parsed = [r for r in records if r["inference"].get("parsed_prediction")]

    print(f"  {'field':40s} {'GT-pos':>8s} {'FN':>5s} {'FN-rate':>8s} "
          f"{'GT-neg':>8s} {'FP':>5s} {'FP-rate':>8s}")
    print(f"  {'-' * 40} {'-' * 8:>8s} {'-' * 5:>5s} {'-' * 8:>8s} "
          f"{'-' * 8:>8s} {'-' * 5:>5s} {'-' * 8:>8s}")

    for (g, p, f) in BOOLEAN_FIELDS:
        tp = tn = fp = fn = 0
        for r in parsed:
            gt = get_nested(r["ground_truth"], (g, p, f))
            pr = get_nested(r["inference"]["parsed_prediction"], (g, p, f))
            if not isinstance(gt, bool) or not isinstance(pr, bool):
                continue
            if gt and pr:   tp += 1
            elif gt and not pr: fn += 1
            elif not gt and pr: fp += 1
            else:           tn += 1
        gt_pos = tp + fn
        gt_neg = tn + fp
        fn_rate = (100 * fn / gt_pos) if gt_pos else 0.0
        fp_rate = (100 * fp / gt_neg) if gt_neg else 0.0
        print(f"  {p + '.' + f:40s} {gt_pos:>8d} {fn:>5d} {fn_rate:>7.1f}% "
              f"{gt_neg:>8d} {fp:>5d} {fp_rate:>7.1f}%")


# =============================================================================
# Section 5 — Final-score metrics on status=='ok' subset
# =============================================================================

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = stats.mean(xs), stats.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def spearman(xs, ys):
    # rank-based pearson
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks
    return pearson(rank(xs), rank(ys))


def section_scores(records: List[dict], label: str):
    print(f"\n{'=' * 72}")
    print(f" [{label}] Score-level metrics  (only status='ok' records)")
    print(f"{'=' * 72}")

    ok = [r for r in records if r["inference"]["status"] == "ok"
          and r.get("computed_scores")]
    if not ok:
        print("  (no 'ok' records; cannot compute score-level metrics)")
        return

    print(f"  Using N = {len(ok)} records")

    gt_scores, pr_scores = [], []
    gt_levels, pr_levels = [], []
    for r in ok:
        gt = r["ground_truth"]["scores"]
        pr = r["computed_scores"]
        gt_scores.append(gt["final_reba_score"])
        pr_scores.append(pr["final_reba_score"])
        gt_levels.append(gt["action_level"]["level"])
        pr_levels.append(pr["action_level"]["level"])

    # final score
    diffs = [p - g for g, p in zip(gt_scores, pr_scores)]
    mae = stats.mean(abs(d) for d in diffs)
    exact = sum(1 for d in diffs if d == 0) / len(diffs)
    mean_bias = stats.mean(diffs)
    print(f"\n  final_reba_score:")
    print(f"    MAE                : {mae:.2f}")
    print(f"    exact match        : {exact:.1%}")
    print(f"    mean bias (pred-gt): {mean_bias:+.2f}    (negative = underestimate)")
    pc = pearson(gt_scores, pr_scores)
    sp = spearman(gt_scores, pr_scores)
    print(f"    Pearson r          : {pc:.3f}" if pc is not None else "    Pearson r          : n/a")
    print(f"    Spearman rho       : {sp:.3f}" if sp is not None else "    Spearman rho       : n/a")

    # distribution of (gt, pred)
    print(f"\n  Samples (gt -> pred):")
    ranges = Counter((g, p) for g, p in zip(gt_scores, pr_scores))
    for (g, p), n in sorted(ranges.items()):
        bar = "█" * n
        print(f"    GT={g:>2d} -> pred={p:>2d}  ({n})  {bar}")

    # action level confusion (levels 0..4)
    print(f"\n  action_level confusion  (rows = GT, cols = pred):")
    levels = sorted(set(gt_levels) | set(pr_levels))
    print(f"    {'gt\\pr':>8s}" + "".join(f"{l:>6d}" for l in levels))
    for g in levels:
        row = [sum(1 for gt_, pr_ in zip(gt_levels, pr_levels) if gt_ == g and pr_ == p_)
               for p_ in levels]
        print(f"    {g:>8d}" + "".join(f"{v:>6d}" for v in row))


# =============================================================================
# Section 6 — per-annotation difficulty (Hamming distance across 17 fields)
# =============================================================================

def section_per_annotation(records: List[dict], label: str, show_top: int = 10):
    print(f"\n{'=' * 72}")
    print(f" [{label}] Per-annotation field error count  (17 fields max)")
    print(f"{'=' * 72}")

    rows = []
    parsed = [r for r in records if r["inference"].get("parsed_prediction")]
    for r in parsed:
        gt = r["ground_truth"]
        pr = r["inference"]["parsed_prediction"]
        errs = 0
        total = 0
        for (g, p, f) in CATEGORICAL_FIELDS + BOOLEAN_FIELDS:
            gv = get_nested(gt, (g, p, f))
            pv = get_nested(pr, (g, p, f))
            if gv is None or pv is None:
                continue
            total += 1
            if gv != pv:
                errs += 1
        rows.append((errs, total, r["annotation_file"],
                     r["inference"]["status"]))

    rows.sort(reverse=True)  # most-wrong first

    print(f"  Top {show_top} hardest (most field errors):")
    print(f"    {'errors':>6s}/{'total':<6s} {'status':18s} annotation")
    for errs, total, fname, status in rows[:show_top]:
        print(f"    {errs:>6d}/{total:<6d} {status:18s} {fname}")

    print(f"\n  Bottom {show_top} easiest (fewest field errors):")
    for errs, total, fname, status in rows[-show_top:]:
        print(f"    {errs:>6d}/{total:<6d} {status:18s} {fname}")

    mean_err = stats.mean(r[0] for r in rows) if rows else 0
    print(f"\n  Mean field-errors per annotation: {mean_err:.2f} / 17")


# =============================================================================
# Method A vs Method B side-by-side (only if both available)
# =============================================================================

def section_method_comparison(records_a: List[dict], records_b: List[dict]):
    print(f"\n{'=' * 72}")
    print(f" Method A vs Method B comparison")
    print(f"{'=' * 72}")

    # Index both by annotation_file
    a_by_ann = {r["annotation_file"]: r for r in records_a}
    b_by_ann = {r["annotation_file"]: r for r in records_b}
    common = sorted(set(a_by_ann) & set(b_by_ann))
    print(f"  N common annotations: {len(common)}")

    # For each field, count: A-only-correct, B-only-correct, both-correct, both-wrong
    print(f"\n  {'field':40s} {'Aonly':>7s} {'Bonly':>7s} {'both✓':>7s} {'both✗':>7s}")
    print(f"  {'-' * 40} {'-' * 7:>7s} {'-' * 7:>7s} {'-' * 7:>7s} {'-' * 7:>7s}")
    for (g, p, f) in CATEGORICAL_FIELDS + BOOLEAN_FIELDS:
        aonly = bonly = both_ok = both_bad = 0
        for ann in common:
            ra = a_by_ann[ann]; rb = b_by_ann[ann]
            if not ra["inference"].get("parsed_prediction"): continue
            if not rb["inference"].get("parsed_prediction"): continue
            gt = get_nested(ra["ground_truth"], (g, p, f))
            pa = get_nested(ra["inference"]["parsed_prediction"], (g, p, f))
            pb = get_nested(rb["inference"]["parsed_prediction"], (g, p, f))
            if gt is None or pa is None or pb is None:
                continue
            a_ok = (pa == gt)
            b_ok = (pb == gt)
            if a_ok and b_ok: both_ok += 1
            elif a_ok and not b_ok: aonly += 1
            elif b_ok and not a_ok: bonly += 1
            else: both_bad += 1
        print(f"  {p + '.' + f:40s} {aonly:>7d} {bonly:>7d} {both_ok:>7d} {both_bad:>7d}")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--method", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    methods = ["A", "B"] if args.method == "both" else [args.method]
    all_records = {}
    for m in methods:
        path = RESULTS_DIR / f"{args.prompt}_method_{m}.jsonl"
        recs = load_jsonl(path)
        if not recs:
            print(f"No results at {path}; skipping method {m}.")
            continue
        all_records[m] = recs

        label = f"{args.prompt}/method-{m}"
        section_outcomes(recs, label)
        section_field_accuracy(recs, label)
        section_confusion(recs, label)
        section_bool_directionality(recs, label)
        section_scores(recs, label)
        section_per_annotation(recs, label)

    if "A" in all_records and "B" in all_records:
        section_method_comparison(all_records["A"], all_records["B"])

    print("\n" + "=" * 72)
    print(" Analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()