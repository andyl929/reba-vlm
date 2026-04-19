"""
inspect_observations.py
-----------------------
Quick inspection of observation field quality in v2 / v2.1 results.

Measures:
  1. Unique ratio: of all observations produced, how many distinct strings?
     A low unique ratio (e.g. 0.3 = 1 in 3 are duplicates) indicates
     boilerplate — the model is generating the same description across
     different videos.
  2. Boilerplate phrase detection: count observations containing
     hedging / non-specific phrases ("relatively", "appears", "standard").
  3. Specificity proxy: count observations containing at least one
     number (degree estimate) or specific spatial term.
  4. Per-part comparison: which body parts get the most boilerplate.

Usage:
    python src/experiments/inspect_observations.py --prompt v2
    python src/experiments/inspect_observations.py --prompt v2_1 --method A
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

PARTS = [
    ("group_a", "trunk"),
    ("group_a", "neck"),
    ("group_a", "legs"),
    ("group_b", "upper_arms"),
    ("group_b", "lower_arms"),
    ("group_b", "wrists"),
    ("context", "load_force"),
    ("context", "coupling"),
]

# Phrases that suggest boilerplate / hedging rather than specific observation
BOILERPLATE_PHRASES = [
    r"\brelatively\b",
    r"\bappears\b",
    r"\bslightly\b",
    r"\bseems\b",
    r"\bnormal\b",
    r"\bstandard\b",
    r"\bnatural(ly)?\b",
    r"\bneutral\b(?!.*°)",  # "neutral" without an angle estimate
    r"\btypical\b",
    r"\bgenerally\b",
    r"\bsomewhat\b",
]

# Presence of these indicates specificity
SPECIFIC_MARKERS = [
    r"\d+\s*°",                      # angle with degree symbol
    r"\d+\s*(deg|degree)",           # "45 deg" or "45 degrees"
    r"~\s*\d+",                      # "~45"
    r"\babout\s+\d+",                # "about 45"
    r"\bapproximately\s+\d+",        # "approximately 45"
    r"\b(near(ly)?|close to|over|past|beyond)\s+\d+",
]


def load_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_observations(records):
    """Return {(part_key): [observations across all records]}"""
    by_part = defaultdict(list)
    for r in records:
        pred = r["inference"].get("parsed_prediction")
        if not pred:
            continue
        for g, p in PARTS:
            obs = (pred.get(g, {}).get(p, {}) or {}).get("observation")
            if isinstance(obs, str) and obs.strip():
                by_part[f"{p}"].append(obs.strip())
    return by_part


def any_pattern(patterns, text):
    return any(re.search(pat, text, re.IGNORECASE) for pat in patterns)


def analyze_observations(by_part):
    print(f"  {'part':22s} {'N':>4s} {'unique':>7s} {'uniq%':>7s} "
          f"{'boiler':>7s} {'specific':>9s}")
    print(f"  {'-'*22} {'-'*4:>4s} {'-'*7:>7s} {'-'*7:>7s} "
          f"{'-'*7:>7s} {'-'*9:>9s}")

    summary = {}
    for part, obs_list in by_part.items():
        n = len(obs_list)
        uniq = len(set(obs_list))
        uniq_pct = 100 * uniq / n if n else 0
        n_boiler = sum(1 for o in obs_list if any_pattern(BOILERPLATE_PHRASES, o))
        n_specific = sum(1 for o in obs_list if any_pattern(SPECIFIC_MARKERS, o))
        print(f"  {part:22s} {n:>4d} {uniq:>7d} {uniq_pct:>6.1f}% "
              f"{n_boiler:>7d} {n_specific:>9d}")
        summary[part] = {
            "n": n, "unique": uniq, "boilerplate": n_boiler,
            "specific": n_specific,
        }
    return summary


def show_examples(by_part, n=5):
    print(f"\n  Most-duplicated observations per part:")
    for part, obs_list in by_part.items():
        counter = Counter(obs_list)
        dupes = [(count, text) for text, count in counter.items() if count > 1]
        dupes.sort(reverse=True)
        if not dupes:
            continue
        print(f"\n  [{part}]")
        for count, text in dupes[:n]:
            # truncate long observations
            short = text if len(text) <= 90 else text[:87] + "..."
            print(f"    ×{count:<2d}  {short}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v2")
    ap.add_argument("--method", choices=["A", "B", "both"], default="both")
    ap.add_argument("--examples", type=int, default=3,
                    help="how many duplicated examples per part to show")
    args = ap.parse_args()

    methods = ["A", "B"] if args.method == "both" else [args.method]

    for m in methods:
        path = RESULTS_DIR / f"{args.prompt}_method_{m}.jsonl"
        records = load_jsonl(path)
        if not records:
            print(f"No records at {path}")
            continue
        print(f"\n{'=' * 70}")
        print(f" {args.prompt} / method {m}   (N={len(records)})")
        print(f"{'=' * 70}")
        by_part = extract_observations(records)
        analyze_observations(by_part)
        if args.examples > 0:
            show_examples(by_part, n=args.examples)


if __name__ == "__main__":
    main()