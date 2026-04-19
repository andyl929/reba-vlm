"""
v1_baseline.py
--------------
Prompt v1: REBA worksheet translated to prompt, minimal scaffolding.

Design principles (per decisions D1-D5):
  - Text faithful to Hignett & McAtamney (2000) Figures 1 & Tables 1-2.
  - JSON output schema enforced with explicit enum values (D2: v1-B),
    because if the model emits free-form strings like "0-20 degrees flex"
    we cannot match them to the annotation schema and the experiment
    collapses into a parse-rate comparison.
  - NO observation guidance, NO worked examples, NO chain-of-thought
    request. Those are the increments that v2 and v3 will add. v1's job
    is to be the floor.
  - The model does NOT compute sub_scores or any lookup-table results.
    It only labels position_class and booleans; our code does the math
    (reba_tables.compute_full_reba).
  - Industry/task context is NOT provided. REBA is designed as a general
    postural tool; injecting context would unfairly strengthen v1.
  - Language: English. REBA terminology is natively English and all
    downstream schema fields are English.

Two input methods share this prompt:
  - Method A: one or more keyframe JPEG(s) prepended; text says
    "Assess the posture shown in the image(s)."
  - Method B: full video prepended + "Key posture moment is at t=X.XXs."
"""

from textwrap import dedent

# =============================================================================
# Enum values (authoritative; imported from reba_tables, kept in sync)
# =============================================================================
# Listed explicitly here (not imported at module top) so the prompt text is
# a single readable string with no hidden dependencies when a reader scans it.
# We assert equality with reba_tables.ENUMS at import time to stay in sync.

_TRUNK_ENUM       = ["upright", "flex_0_to_20_or_ext_0_to_20",
                     "flex_20_to_60_or_ext_gt_20", "flex_gt_60"]
_NECK_ENUM        = ["flex_0_to_20", "flex_gt_20_or_ext"]
_LEGS_ENUM        = ["bilateral_weight_bearing_or_sitting",
                     "unilateral_feather_or_unstable"]
_UPPER_ARMS_ENUM  = ["ext_20_to_flex_20", "ext_gt_20_or_flex_20_to_45",
                     "flex_45_to_90", "flex_gt_90"]
_LOWER_ARMS_ENUM  = ["flex_60_to_100", "flex_lt_60_or_gt_100"]
_WRISTS_ENUM      = ["flex_ext_0_to_15", "flex_ext_gt_15"]
_LOAD_ENUM        = ["lt_5_kg", "5_to_10_kg", "gt_10_kg"]
_COUPLING_ENUM    = ["good_power_grip", "fair_acceptable_hold",
                     "poor_not_acceptable", "unacceptable_unsafe"]


def _enum_block(name: str, values: list) -> str:
    """Render an enum list for the prompt."""
    return f'"{name}": one of {values}'


# =============================================================================
# The prompt text (v1 baseline)
# =============================================================================

V1_BASELINE_TEXT = dedent("""\
    You are assessing a worker's posture using Rapid Entire Body Assessment
    (REBA), per Hignett & McAtamney (2000).

    REBA is structured into Group A (trunk, neck, legs) and Group B (upper
    arms, lower arms, wrists), plus Load/Force and Coupling context.

    === Group A definitions ===

    Trunk position:
      - Upright
      - 0°-20° flexion, or 0°-20° extension
      - 20°-60° flexion, or >20° extension
      - >60° flexion
    Adjustment: trunk is twisted or side-flexed (yes/no).

    Neck position:
      - 0°-20° flexion
      - >20° flexion, or in extension
    Adjustment: neck is twisted or side-flexed (yes/no).

    Legs position:
      - Bilateral weight bearing, walking, or sitting
      - Unilateral weight bearing, feather weight bearing, or unstable
    Adjustments: one or both knees in 30°-60° flexion (yes/no);
                 one or both knees in >60° flexion -- not for sitting (yes/no).

    === Group B definitions ===

    Upper arm position (analyze the dominant/more-loaded arm):
      - 20° extension to 20° flexion
      - >20° extension, or 20°-45° flexion
      - 45°-90° flexion
      - >90° flexion
    Adjustments: arm is abducted or rotated (yes/no);
                 shoulder is raised (yes/no);
                 leaning, supporting weight of arm, or posture is gravity-
                 assisted (yes/no).

    Lower arm position:
      - 60°-100° flexion
      - <60° flexion, or >100° flexion

    Wrist position:
      - 0°-15° flexion/extension
      - >15° flexion/extension
    Adjustment: wrist is deviated or twisted (yes/no).

    === Load/Force ===
      - < 5 kg
      - 5-10 kg
      - > 10 kg
    Adjustment: shock or rapid build-up of force (yes/no).

    === Coupling (grip quality) ===
      - Good: well-fitting handle, mid-range power grip
      - Fair: hand hold acceptable but not ideal, or coupling via another body part
      - Poor: hand hold not acceptable though possible
      - Unacceptable: awkward, unsafe grip, no handles

    ===============================================================
    OUTPUT
    ===============================================================

    Return ONLY a single JSON object matching exactly this schema. Do not
    include any text before or after the JSON. Do not use Markdown code
    fences. Every field is required.

    The string values for position_class and category_class MUST be taken
    verbatim from the allowed enumerations below -- no variations, no free
    text.

    {{
      "group_a": {{
        "trunk": {{
          {trunk_enum},
          "is_twisted_sidebent": <true|false>
        }},
        "neck": {{
          {neck_enum},
          "is_twisted_sidebent": <true|false>
        }},
        "legs": {{
          {legs_enum},
          "knee_30_60_flexion": <true|false>,
          "knee_gt_60_flexion": <true|false>
        }}
      }},
      "group_b": {{
        "upper_arms": {{
          {upper_arms_enum},
          "is_abducted_rotated": <true|false>,
          "is_shoulder_raised": <true|false>,
          "is_gravity_assisted": <true|false>
        }},
        "lower_arms": {{
          {lower_arms_enum}
        }},
        "wrists": {{
          {wrists_enum},
          "is_deviated_twisted": <true|false>
        }}
      }},
      "context": {{
        "load_force": {{
          {load_enum},
          "has_shock": <true|false>
        }},
        "coupling": {{
          {coupling_enum}
        }}
      }}
    }}
    """).format(
        trunk_enum=_enum_block("position_class", _TRUNK_ENUM),
        neck_enum=_enum_block("position_class", _NECK_ENUM),
        legs_enum=_enum_block("position_class", _LEGS_ENUM),
        upper_arms_enum=_enum_block("position_class", _UPPER_ARMS_ENUM),
        lower_arms_enum=_enum_block("position_class", _LOWER_ARMS_ENUM),
        wrists_enum=_enum_block("position_class", _WRISTS_ENUM),
        load_enum=_enum_block("category_class", _LOAD_ENUM),
        coupling_enum=_enum_block("category_class", _COUPLING_ENUM),
    )


# Short trailers describing the visual input (method-specific)
METHOD_A_TRAILER = "\nAssess the posture shown in the image(s) above."
METHOD_B_TRAILER = (
    "\nAssess the posture in the video above. "
    "The key moment to evaluate is at t={timestamp:.2f} seconds."
)


def build_prompt(method: str, timestamp: float = None) -> str:
    """
    Build the full v1 user-turn text given the input method.

    Parameters
    ----------
    method : 'A' (images) or 'B' (video)
    timestamp : required for method B; the keyframe time in seconds.
    """
    if method == "A":
        return V1_BASELINE_TEXT + METHOD_A_TRAILER
    if method == "B":
        if timestamp is None:
            raise ValueError("Method B requires a timestamp (seconds).")
        return V1_BASELINE_TEXT + METHOD_B_TRAILER.format(timestamp=timestamp)
    raise ValueError(f"Unknown method: {method!r}. Use 'A' or 'B'.")


# =============================================================================
# Sanity check against reba_tables.ENUMS at import time
# =============================================================================

def _verify_enums_synced():
    """Fail fast if we drift from the single source of truth."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from reba_tables import ENUMS as _ENUMS

    checks = [
        ("trunk.position_class",       _TRUNK_ENUM),
        ("neck.position_class",        _NECK_ENUM),
        ("legs.position_class",        _LEGS_ENUM),
        ("upper_arms.position_class",  _UPPER_ARMS_ENUM),
        ("lower_arms.position_class",  _LOWER_ARMS_ENUM),
        ("wrists.position_class",      _WRISTS_ENUM),
        ("load_force.category_class",  _LOAD_ENUM),
        ("coupling.category_class",    _COUPLING_ENUM),
    ]
    for key, local in checks:
        canonical = _ENUMS[key]
        if list(local) != list(canonical):
            raise RuntimeError(
                f"Prompt enum out of sync with reba_tables.ENUMS for {key}:\n"
                f"  prompt:  {local}\n"
                f"  tables:  {canonical}"
            )


_verify_enums_synced()


if __name__ == "__main__":
    print("=" * 70)
    print("METHOD A prompt:")
    print("=" * 70)
    print(build_prompt("A"))
    print("\n" + "=" * 70)
    print("METHOD B prompt (timestamp=3.47):")
    print("=" * 70)
    print(build_prompt("B", timestamp=3.47))
    print("\n✓ Enums synced with reba_tables.")