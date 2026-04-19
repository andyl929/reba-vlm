"""
reba_tables.py
--------------
Single source of truth for REBA scoring logic.

Given the set of categorical fields that match our annotation schema
(position_class + booleans for each body part + load/coupling categories),
this module computes every sub-score and the final REBA score using:
  - per-part sub_score = base(position_class) + signed boolean adjustments
  - Table A: (trunk, neck, legs) -> table_a_score
  - Load adjustment: score_a = table_a_score + load_sub_score
  - Table B: (upper_arms, lower_arms, wrists) -> table_b_score
  - Coupling adjustment: score_b = table_b_score + coupling_sub_score
  - Table C: (score_a, score_b) -> score_c
  - Activity adjustment: final_reba_score = score_c + activity_score  (we fix activity=0)

Lookup tables TABLE_A, TABLE_B, TABLE_C copied verbatim from the verified
Streamlit annotation tool the project lead (Andy Li) used to produce
ground-truth JSONs. This guarantees consistency between annotations and
the computed references we use to score model predictions.

References:
  Hignett & McAtamney (2000). Rapid Entire Body Assessment (REBA).
  Applied Ergonomics 31(2), 201-205.
"""

from typing import Dict, Optional, Tuple, Any

# =============================================================================
# Lookup tables (from the annotation tool's TABLE_A/B/C, verified)
# =============================================================================

# TABLE_A[trunk-1][neck-1][legs-1]  -- trunk 1..5, neck 1..3, legs 1..4
TABLE_A = [
    [[1, 2, 3, 4], [1, 2, 3, 4], [3, 3, 5, 6]],
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
    [[2, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[3, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[4, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
]

# TABLE_B[upper_arms-1][lower_arms-1][wrists-1]  -- ua 1..6, la 1..2, wrists 1..3
TABLE_B = [
    [[1, 2, 2], [1, 2, 3]],
    [[1, 2, 3], [2, 3, 4]],
    [[3, 4, 5], [4, 5, 5]],
    [[4, 5, 5], [5, 6, 7]],
    [[6, 7, 8], [7, 8, 8]],
    [[7, 8, 8], [8, 9, 9]],
]

# TABLE_C[score_a-1][score_b-1]  -- both 1..12
TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 8, 8, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 8, 8, 9, 9, 9, 9],
    [4, 4, 4, 5, 6, 8, 8, 9, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]

# =============================================================================
# Categorical label -> base score
# =============================================================================

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
    "ext_20_to_flex_20": 1,
    "ext_gt_20_or_flex_20_to_45": 2,
    "flex_45_to_90": 3,
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

# Canonical exposed enumerations for prompt-building / validation elsewhere.
ENUMS = {
    "trunk.position_class": list(TRUNK_BASE.keys()),
    "neck.position_class": list(NECK_BASE.keys()),
    "legs.position_class": list(LEGS_BASE.keys()),
    "upper_arms.position_class": list(UPPER_ARMS_BASE.keys()),
    "lower_arms.position_class": list(LOWER_ARMS_BASE.keys()),
    "wrists.position_class": list(WRISTS_BASE.keys()),
    "load_force.category_class": list(LOAD_BASE.keys()),
    "coupling.category_class": list(COUPLING_BASE.keys()),
}


# =============================================================================
# Sub-score computers  (each returns int or None if inputs are invalid)
# =============================================================================

def _get(d: Dict, key: str, default: Any = None):
    """Safe .get for dict-ish inputs."""
    if d is None:
        return default
    return d.get(key, default)


def compute_trunk(p: Dict) -> Optional[int]:
    base = TRUNK_BASE.get(_get(p, "position_class"))
    if base is None:
        return None
    return base + (1 if _get(p, "is_twisted_sidebent", False) else 0)


def compute_neck(p: Dict) -> Optional[int]:
    base = NECK_BASE.get(_get(p, "position_class"))
    if base is None:
        return None
    return base + (1 if _get(p, "is_twisted_sidebent", False) else 0)


def compute_legs(p: Dict) -> Optional[int]:
    base = LEGS_BASE.get(_get(p, "position_class"))
    if base is None:
        return None
    s = base
    if _get(p, "knee_30_60_flexion", False):
        s += 1
    if _get(p, "knee_gt_60_flexion", False):
        s += 2
    return s


def compute_upper_arms(p: Dict) -> Optional[int]:
    base = UPPER_ARMS_BASE.get(_get(p, "position_class"))
    if base is None:
        return None
    s = base
    if _get(p, "is_abducted_rotated", False):
        s += 1
    if _get(p, "is_shoulder_raised", False):
        s += 1
    if _get(p, "is_gravity_assisted", False):
        s -= 1
    return s


def compute_lower_arms(p: Dict) -> Optional[int]:
    return LOWER_ARMS_BASE.get(_get(p, "position_class"))


def compute_wrists(p: Dict) -> Optional[int]:
    base = WRISTS_BASE.get(_get(p, "position_class"))
    if base is None:
        return None
    return base + (1 if _get(p, "is_deviated_twisted", False) else 0)


def compute_load(p: Dict) -> Optional[int]:
    base = LOAD_BASE.get(_get(p, "category_class"))
    if base is None:
        return None
    return base + (1 if _get(p, "has_shock", False) else 0)


def compute_coupling(p: Dict) -> Optional[int]:
    return COUPLING_BASE.get(_get(p, "category_class"))


# =============================================================================
# Table lookups
# =============================================================================

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def lookup_table_a(trunk: int, neck: int, legs: int) -> int:
    """trunk: 1..5, neck: 1..3, legs: 1..4  (values outside range are clamped)."""
    t = _clamp(trunk, 1, 5) - 1
    n = _clamp(neck, 1, 3) - 1
    l = _clamp(legs, 1, 4) - 1
    return TABLE_A[t][n][l]


def lookup_table_b(upper_arms: int, lower_arms: int, wrists: int) -> int:
    """upper_arms: 1..6, lower_arms: 1..2, wrists: 1..3."""
    ua = _clamp(upper_arms, 1, 6) - 1
    la = _clamp(lower_arms, 1, 2) - 1
    w = _clamp(wrists, 1, 3) - 1
    return TABLE_B[ua][la][w]


def lookup_table_c(score_a: int, score_b: int) -> int:
    """Both 1..12."""
    a = _clamp(score_a, 1, 12) - 1
    b = _clamp(score_b, 1, 12) - 1
    return TABLE_C[a][b]


# =============================================================================
# Action level (per Table 4 of the REBA paper)
# =============================================================================

def action_level(final_score: int) -> Dict[str, Any]:
    if final_score == 1:
        return {"level": 0, "risk": "Negligible", "action_required": "None necessary"}
    if 2 <= final_score <= 3:
        return {"level": 1, "risk": "Low", "action_required": "May be necessary"}
    if 4 <= final_score <= 7:
        return {"level": 2, "risk": "Medium", "action_required": "Necessary"}
    if 8 <= final_score <= 10:
        return {"level": 3, "risk": "High", "action_required": "Necessary soon"}
    if 11 <= final_score <= 15:
        return {"level": 4, "risk": "Very high", "action_required": "Necessary NOW"}
    return {"level": -1, "risk": "Unknown", "action_required": "Unknown"}


# =============================================================================
# Full pipeline: annotation-shaped dict -> fully-computed scores
# =============================================================================

class RebaComputationError(Exception):
    """Raised when a required categorical input is missing or unrecognized."""


def compute_full_reba(annotation: Dict, activity_score: int = 0,
                      strict: bool = True) -> Dict[str, Any]:
    """
    Given an annotation-shaped dict (may be a ground-truth JSON or a model
    prediction with the same schema), compute every score from scratch.

    Parameters
    ----------
    annotation : dict with keys 'group_a', 'group_b', 'context'
        (meta_data and 'scores' are ignored)
    activity_score : int, default 0
        Per project decision, activity is not predicted (see handoff Sec 2).
    strict : bool, default True
        If True, raise RebaComputationError on any unrecognized/missing
        categorical field. If False, return None for any unresolvable part
        and propagate None through the table lookups (will clamp -> 1).

    Returns
    -------
    dict with the same 'scores' substructure as the ground-truth JSON, plus
    a per-part 'sub_score' map under 'sub_scores' for debugging.
    """
    ga = annotation.get("group_a", {}) or {}
    gb = annotation.get("group_b", {}) or {}
    ctx = annotation.get("context", {}) or {}

    parts = {
        "trunk": compute_trunk(ga.get("trunk", {})),
        "neck": compute_neck(ga.get("neck", {})),
        "legs": compute_legs(ga.get("legs", {})),
        "upper_arms": compute_upper_arms(gb.get("upper_arms", {})),
        "lower_arms": compute_lower_arms(gb.get("lower_arms", {})),
        "wrists": compute_wrists(gb.get("wrists", {})),
        "load_force": compute_load(ctx.get("load_force", {})),
        "coupling": compute_coupling(ctx.get("coupling", {})),
    }

    if strict:
        missing = [k for k, v in parts.items() if v is None]
        if missing:
            raise RebaComputationError(
                f"Could not compute sub-score for: {missing}. "
                f"Check that position_class / category_class values are in ENUMS."
            )

    # Default Nones to 1 (lowest) so table lookups don't crash in non-strict mode
    p = {k: (1 if v is None else v) for k, v in parts.items()}

    table_a = lookup_table_a(p["trunk"], p["neck"], p["legs"])
    score_a = table_a + p["load_force"]

    table_b = lookup_table_b(p["upper_arms"], p["lower_arms"], p["wrists"])
    score_b = table_b + p["coupling"]

    score_c = lookup_table_c(score_a, score_b)
    final = score_c + activity_score
    final = _clamp(final, 1, 15)

    return {
        "sub_scores": parts,
        "scores": {
            "table_a": table_a,
            "score_a": score_a,
            "table_b": table_b,
            "score_b": score_b,
            "score_c": score_c,
            "activity_score": activity_score,
            "final_reba_score": final,
            "action_level": action_level(final),
        },
    }


# =============================================================================
# Smoke test (runs when executed directly)
# =============================================================================

if __name__ == "__main__":
    # Replay the worked example from Hignett & McAtamney (2000), p.204:
    # Trunk >60° flexion + side-flexed -> 4+1=5
    # Neck in extension -> 2
    # Legs bilateral weight bearing + knees >60° flex -> 1+2=3
    # Upper arm 45-90° flex + abducted (+1) + gravity-assisted (-1) -> 3+1-1=3
    # Lower arm <60° flex -> 2
    # Wrist 0-15° flex -> 1
    # Load < 5 kg -> 0
    # Coupling Fair -> 1
    # Activity +1 (large range posture change)
    # Expected: Score A = 8, Score B = 5, Score C = 10, final = 11
    demo = {
        "group_a": {
            "trunk": {"position_class": "flex_gt_60", "is_twisted_sidebent": True},
            "neck":  {"position_class": "flex_gt_20_or_ext", "is_twisted_sidebent": False},
            "legs":  {"position_class": "bilateral_weight_bearing_or_sitting",
                      "knee_30_60_flexion": False, "knee_gt_60_flexion": True},
        },
        "group_b": {
            "upper_arms": {"position_class": "flex_45_to_90",
                           "is_abducted_rotated": True,
                           "is_shoulder_raised": False,
                           "is_gravity_assisted": True},
            "lower_arms": {"position_class": "flex_lt_60_or_gt_100"},
            "wrists":     {"position_class": "flex_ext_0_to_15", "is_deviated_twisted": False},
        },
        "context": {
            "load_force": {"category_class": "lt_5_kg", "has_shock": False},
            "coupling":   {"category_class": "fair_acceptable_hold"},
        },
    }
    r = compute_full_reba(demo, activity_score=1)
    print("Worked example from Hignett & McAtamney (2000):")
    print(f"  sub_scores      : {r['sub_scores']}")
    print(f"  table_a         : {r['scores']['table_a']}")
    print(f"  score_a         : {r['scores']['score_a']}     (expected 8)")
    print(f"  table_b         : {r['scores']['table_b']}")
    print(f"  score_b         : {r['scores']['score_b']}     (expected 5)")
    print(f"  score_c         : {r['scores']['score_c']}    (expected 10)")
    print(f"  final_reba_score: {r['scores']['final_reba_score']}    (expected 11)")
    print(f"  action_level    : {r['scores']['action_level']}")
    assert r["scores"]["score_a"] == 8
    assert r["scores"]["score_b"] == 5
    assert r["scores"]["score_c"] == 10
    assert r["scores"]["final_reba_score"] == 11
    print("\n✓ Worked example matches paper.")