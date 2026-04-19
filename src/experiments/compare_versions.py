"""
compare_versions.py
-------------------
Side-by-side comparison of two prompt versions on the same (method) subset.
Loads both versions' JSONL result files and prints matched metrics tables.

Usage:
    python src/experiments/compare_versions.py                    # v1 vs v2, both methods
    python src/experiments/compare_versions.py --methods A        # just method A
    python src/experiments/compare_versions.py --from v1 --to v2  # explicit

Report includes:
  - Parse / schema outcome deltas
  - Per-field accuracy deltas (which fields improved, which regressed)
  - final_reba_score metric deltas (MAE, correlation, exact match)
  - action_level confusion delta
  - Per-annotation: was each sample improved, regressed, or unchanged
"""

import argparse
import json
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reba_tables import ENUMS  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"

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


def get_nested(d: dict, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


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
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    return pearson(rank(xs), rank(ys))


# =============================================================================
# Individual sections
# =============================================================================

def section_outcomes(label_from: str, label_to: str,
                     recs_from: List[dict], recs_to: List[dict]):
    print(f"\n{'=' * 72}")
    print(f" Parse/schema outcomes: {label_from}  →  {label_to}")
    print(f"{'=' * 72}")
    cf = Counter(r["inference"]["status"] for r in recs_from)
    ct = Counter(r["inference"]["status"] for r in recs_to)
    print(f"  {'status':20s} {label_from:>8s} {label_to:>8s} {'Δ':>8s}")
    for s in ("ok", "schema_invalid", "parse_failed", "http_error"):
        f, t = cf.get(s, 0), ct.get(s, 0)
        arrow = "+" if t > f else ("-" if t < f else " ")
        print(f"  {s:20s} {f:>8d} {t:>8d} {arrow}{abs(t-f):>7d}")


def field_acc(records: List[dict], path):
    correct = total = 0
    for r in records:
        pr = r["inference"].get("parsed_prediction")
        if not pr:
            continue
        gt_v = get_nested(r["ground_truth"], path)
        pr_v = get_nested(pr, path)
        if gt_v is None or pr_v is None:
            continue
        total += 1
        if gt_v == pr_v:
            correct += 1
    return correct, total


def section_field_accuracy(label_from: str, label_to: str,
                           recs_from: List[dict], recs_to: List[dict]):
    print(f"\n{'=' * 72}")
    print(f" Per-field accuracy  (uses all records with parsed_prediction)")
    print(f"{'=' * 72}")
    print(f"  {'field':38s}  {label_from:>8s}   {label_to:>8s}   {'Δ pct-pts':>10s}   note")
    print(f"  {'-'*38}  {'-'*8:>8s}   {'-'*8:>8s}   {'-'*10:>10s}   ----")

    deltas = []
    for (g, p, f) in CATEGORICAL_FIELDS + BOOLEAN_FIELDS:
        cf, tf = field_acc(recs_from, (g, p, f))
        ct, tt = field_acc(recs_to, (g, p, f))
        af = 100 * cf / tf if tf else 0
        at = 100 * ct / tt if tt else 0
        d = at - af
        note = "▲" if d > 3 else ("▼" if d < -3 else " ")
        print(f"  {p + '.' + f:38s}  {af:>7.1f}%   {at:>7.1f}%   {d:>+9.1f}  {note}")
        deltas.append((f"{p}.{f}", d))

    improved = sum(1 for _, d in deltas if d > 3)
    regressed = sum(1 for _, d in deltas if d < -3)
    flat = len(deltas) - improved - regressed
    print(f"\n  Summary: {improved} fields improved ≥3pp, "
          f"{regressed} regressed ≥3pp, {flat} unchanged")

    cat_f = sum(field_acc(recs_from, fld)[0] for fld in CATEGORICAL_FIELDS)
    cat_t = sum(field_acc(recs_to,   fld)[0] for fld in CATEGORICAL_FIELDS)
    cat_n_f = sum(field_acc(recs_from, fld)[1] for fld in CATEGORICAL_FIELDS)
    cat_n_t = sum(field_acc(recs_to,   fld)[1] for fld in CATEGORICAL_FIELDS)
    bool_f = sum(field_acc(recs_from, fld)[0] for fld in BOOLEAN_FIELDS)
    bool_t = sum(field_acc(recs_to,   fld)[0] for fld in BOOLEAN_FIELDS)
    bool_n_f = sum(field_acc(recs_from, fld)[1] for fld in BOOLEAN_FIELDS)
    bool_n_t = sum(field_acc(recs_to,   fld)[1] for fld in BOOLEAN_FIELDS)

    def pct(c, n): return 100 * c / n if n else 0
    print(f"\n  {'[categorical mean]':38s}  {pct(cat_f, cat_n_f):>7.1f}%   "
          f"{pct(cat_t, cat_n_t):>7.1f}%   "
          f"{pct(cat_t, cat_n_t) - pct(cat_f, cat_n_f):>+9.1f}")
    print(f"  {'[boolean mean]':38s}  {pct(bool_f, bool_n_f):>7.1f}%   "
          f"{pct(bool_t, bool_n_t):>7.1f}%   "
          f"{pct(bool_t, bool_n_t) - pct(bool_f, bool_n_f):>+9.1f}")
    total_f = cat_f + bool_f
    total_t = cat_t + bool_t
    total_n_f = cat_n_f + bool_n_f
    total_n_t = cat_n_t + bool_n_t
    print(f"  {'[all fields mean]':38s}  {pct(total_f, total_n_f):>7.1f}%   "
          f"{pct(total_t, total_n_t):>7.1f}%   "
          f"{pct(total_t, total_n_t) - pct(total_f, total_n_f):>+9.1f}")


def section_scores(label_from: str, label_to: str,
                   recs_from: List[dict], recs_to: List[dict]):
    print(f"\n{'=' * 72}")
    print(f" Final-score metrics  (only status='ok' records)")
    print(f"{'=' * 72}")

    def collect(records):
        ok = [r for r in records if r["inference"]["status"] == "ok"
              and r.get("computed_scores")]
        gt = [r["ground_truth"]["scores"]["final_reba_score"] for r in ok]
        pr = [r["computed_scores"]["final_reba_score"] for r in ok]
        return ok, gt, pr

    ok_f, gt_f, pr_f = collect(recs_from)
    ok_t, gt_t, pr_t = collect(recs_to)

    def metrics(gt, pr):
        if not gt:
            return None
        diffs = [p - g for g, p in zip(gt, pr)]
        mae = stats.mean(abs(d) for d in diffs)
        exact = 100 * sum(1 for d in diffs if d == 0) / len(diffs)
        bias = stats.mean(diffs)
        return {
            "N": len(gt), "MAE": mae, "exact": exact,
            "bias": bias,
            "pearson": pearson(gt, pr),
            "spearman": spearman(gt, pr),
        }

    mf = metrics(gt_f, pr_f)
    mt = metrics(gt_t, pr_t)

    print(f"  {'metric':22s} {label_from:>10s} {label_to:>10s}    Δ")
    print(f"  {'-'*22} {'-'*10:>10s} {'-'*10:>10s}    -----")

    def fmt(v):
        return f"{v:.3f}" if v is not None else "  n/a"

    print(f"  {'N (ok)':22s} {(mf or {}).get('N', 0):>10d} "
          f"{(mt or {}).get('N', 0):>10d}    "
          f"{(mt or {}).get('N', 0) - (mf or {}).get('N', 0):+d}")

    if not mf or not mt:
        print("  (one side has no ok samples; remaining metrics skipped)")
        return

    print(f"  {'MAE (lower better)':22s} {mf['MAE']:>10.2f} {mt['MAE']:>10.2f}    "
          f"{mt['MAE'] - mf['MAE']:+.2f}")
    print(f"  {'exact match %':22s} {mf['exact']:>10.1f} {mt['exact']:>10.1f}    "
          f"{mt['exact'] - mf['exact']:+.1f}")
    print(f"  {'mean bias':22s} {mf['bias']:>+10.2f} {mt['bias']:>+10.2f}    "
          f"{mt['bias'] - mf['bias']:+.2f}")
    print(f"  {'Pearson r':22s}   {fmt(mf['pearson'])}     {fmt(mt['pearson'])}      "
          f"{(mt['pearson'] or 0) - (mf['pearson'] or 0):+.3f}")
    print(f"  {'Spearman rho':22s}   {fmt(mf['spearman'])}     {fmt(mt['spearman'])}      "
          f"{(mt['spearman'] or 0) - (mf['spearman'] or 0):+.3f}")


def section_per_annotation_delta(label_from: str, label_to: str,
                                 recs_from: List[dict], recs_to: List[dict]):
    """Which annotations got better / worse in field-error count?"""
    print(f"\n{'=' * 72}")
    print(f" Per-annotation change  (field errors out of 17; lower = better)")
    print(f"{'=' * 72}")

    def errs_by_ann(records):
        out = {}
        for r in records:
            pr = r["inference"].get("parsed_prediction")
            if not pr:
                continue
            errs = 0
            for fld in CATEGORICAL_FIELDS + BOOLEAN_FIELDS:
                gv = get_nested(r["ground_truth"], fld)
                pv = get_nested(pr, fld)
                if gv is None or pv is None:
                    continue
                if gv != pv:
                    errs += 1
            out[r["annotation_file"]] = errs
        return out

    ef = errs_by_ann(recs_from)
    et = errs_by_ann(recs_to)
    common = sorted(set(ef) & set(et))

    rows = [(et[a] - ef[a], ef[a], et[a], a) for a in common]
    rows.sort()  # most improved first (most negative delta)

    improved = sum(1 for d, *_ in rows if d < 0)
    same = sum(1 for d, *_ in rows if d == 0)
    regressed = sum(1 for d, *_ in rows if d > 0)
    print(f"  Improved: {improved}    Unchanged: {same}    Regressed: {regressed}    "
          f"(N = {len(common)})")

    n_show = 10
    print(f"\n  Top {n_show} most improved:")
    for d, ef_, et_, a in rows[:n_show]:
        print(f"    {a:32s}  {ef_:>2d}  →  {et_:>2d}   (Δ {d:+d})")

    print(f"\n  Top {n_show} most regressed:")
    for d, ef_, et_, a in rows[-n_show:][::-1]:
        print(f"    {a:32s}  {ef_:>2d}  →  {et_:>2d}   (Δ {d:+d})")

    mean_f = stats.mean(ef[a] for a in common)
    mean_t = stats.mean(et[a] for a in common)
    print(f"\n  Mean field-errors/annotation: "
          f"{mean_f:.2f} → {mean_t:.2f}   (Δ {mean_t - mean_f:+.2f})")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="v_from", default="v1",
                    help="baseline prompt version (default v1)")
    ap.add_argument("--to", dest="v_to", default="v2",
                    help="new prompt version to compare (default v2)")
    ap.add_argument("--methods", default="both",
                    choices=["A", "B", "both"])
    args = ap.parse_args()

    methods = ["A", "B"] if args.methods == "both" else [args.methods]

    for m in methods:
        path_f = RESULTS_DIR / f"{args.v_from}_method_{m}.jsonl"
        path_t = RESULTS_DIR / f"{args.v_to}_method_{m}.jsonl"
        rf = load_jsonl(path_f)
        rt = load_jsonl(path_t)
        if not rf or not rt:
            print(f"\nSkipping method {m}: missing results "
                  f"({path_f.name} has {len(rf)}, {path_t.name} has {len(rt)})")
            continue

        print(f"\n{'#' * 72}")
        print(f"# {args.v_from}  vs  {args.v_to}   |   Method {m}   |   "
              f"N_from={len(rf)}  N_to={len(rt)}")
        print(f"{'#' * 72}")

        section_outcomes(args.v_from, args.v_to, rf, rt)
        section_field_accuracy(args.v_from, args.v_to, rf, rt)
        section_scores(args.v_from, args.v_to, rf, rt)
        section_per_annotation_delta(args.v_from, args.v_to, rf, rt)

    print("\n" + "=" * 72)
    print(" Comparison complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()