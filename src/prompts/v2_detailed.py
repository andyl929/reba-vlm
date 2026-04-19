"""
v2_detailed.py
--------------
Prompt v2: REBA scoring with per-part observation guidance, corrected angular
reference frames, and explicit adjustment checklists. Addresses the five
systematic failure modes of v1 identified from the v1 experiment:

  E1. Angular reference frame (upper arms): measured RELATIVE TO THE TRUNK,
      not relative to vertical. This is the single most consequential
      change; v1 misclassified trunk-bent-forward cases because the arm
      looked vertical from the camera yet was "neutral" relative to a
      forward-bent torso.
  E2. Enum anti-abbreviation: every enum value is shown next to its plain
      language meaning, with a direct warning NOT to shorten names.
      Targets v1's single failure mode driving ALL schema_invalid cases
      ("flex_20_to_45" instead of "ext_gt_20_or_flex_20_to_45").
  E3. Observation-before-classification: each part gets an 'observation'
      field asking for one sentence describing what the model sees before
      picking the category. This is a structured visual-to-text step.
      The observation field is scored in logs but NOT in field accuracy
      (does not appear in the annotation schema).
  E4. Active adjustment checklist: explicit prompts to LOOK FOR rotation,
      side-bend, abduction, shoulder raise. Targets v1's 75-92% FN rates
      on these boolean fields (v1 defaults them to false).
  E5. Counter to regression-to-the-mean: warning that extreme postures
      (trunk > 60°, arm above head) should be scored as such, not softened.
      Targets v1 predicting "flex_20_to_60" for 11/15 of GT=flex_gt_60.
  E6. Calibrated wrist guidance: the wrist field is partially subjective
      even for human annotators; guidance here is softer. When deviation
      is not visually clear, a false answer is acceptable.

Output schema is identical to v1 plus a new `observation` field inside each
part. The observation field is discarded during evaluation (analyze_results.py
only inspects the fields listed in CATEGORICAL_FIELDS / BOOLEAN_FIELDS).
"""

from textwrap import dedent


# =============================================================================
# The prompt text (v2 detailed)
# =============================================================================

V2_DETAILED_TEXT = dedent("""\
    You are assessing a worker's posture using Rapid Entire Body Assessment
    (REBA), per Hignett & McAtamney (2000).

    You will score 8 body / context components. For each one, you first
    describe what you see in one sentence ("observation"), then assign the
    categorical class, then answer any yes/no adjustment questions.

    ============================================================
    IMPORTANT VIEWING PRINCIPLES
    ============================================================

    1. ANGULAR REFERENCE FRAMES
       - TRUNK and NECK angles are measured from the UPRIGHT body position.
         Bending forward or backward from standing straight is flexion/
         extension.
       - UPPER ARM angle is measured RELATIVE TO THE TRUNK, not relative
         to vertical or to the ground. When a worker bends forward at the
         waist, an arm hanging straight down looks vertical from the
         camera, yet relative to the bent trunk it is at a LARGE flexion
         angle. Always mentally align with the torso line (hip-to-
         shoulder) first, then read arm angle from there.
       - LOWER ARM angle is the elbow joint angle (flexion between upper
         and lower arm).
       - WRIST angle is the deviation of the hand from the forearm line.

    2. EXTREME POSTURES ARE COMMON
       Many worker postures in this dataset are extreme: trunks bent past
       60°, arms raised overhead, knees in deep flexion. Do NOT default
       to "moderate" categories when the posture is clearly extreme.
       Trust what you see. Under-scoring by choosing safer middle
       categories is the single biggest failure mode to avoid.

    3. ACTIVELY LOOK FOR ADJUSTMENTS
       For every boolean adjustment, deliberately check for it. Do not
       answer "false" by default — "false" is only correct when you have
       actually looked and confirmed the deviation is absent.

    4. WHEN THE ANNOTATED ARM IS UNCLEAR
       Assess the dominant / more heavily loaded arm. If both arms are
       similarly loaded, pick the one in the more extreme posture.

    ============================================================
    COMPONENT DEFINITIONS WITH EXACT ENUM VALUES
    ============================================================

    The position_class and category_class strings below MUST be copied
    VERBATIM into your JSON output. Do not shorten them. In particular,
    several enums contain BOTH a flexion and an extension clause joined
    by "_or_" — the full string is required even when only one clause
    applies to this worker.

    ------------------------------------------------------------
    TRUNK
    ------------------------------------------------------------
    Observation: describe the torso's lean direction and roughly how many
    degrees from upright.

    position_class must be exactly one of:
      - "upright"                                (standing straight, < ~5° lean)
      - "flex_0_to_20_or_ext_0_to_20"            (slight forward or backward lean, up to 20°)
      - "flex_20_to_60_or_ext_gt_20"             (clear forward lean 20-60°, OR any backward lean > 20°)
      - "flex_gt_60"                             (deeply bent forward, more than 60°)

    Check explicitly:
      is_twisted_sidebent: is the torso rotated (one shoulder clearly
        further from the camera than the other) or leaning to the side
        (spine not aligned vertically with pelvis)?

    ------------------------------------------------------------
    NECK
    ------------------------------------------------------------
    Observation: describe head direction and roughly the flexion/extension angle.

    position_class must be exactly one of:
      - "flex_0_to_20"                           (head nearly upright to slightly down, up to 20°)
      - "flex_gt_20_or_ext"                      (head clearly bent forward > 20°, OR tilted back)

    Check explicitly:
      is_twisted_sidebent: is the head turned to one side, or tilted
        toward one shoulder?

    ------------------------------------------------------------
    LEGS
    ------------------------------------------------------------
    Observation: describe stance (standing, walking, sitting, kneeling,
    one-legged, balancing) and knee position.

    position_class must be exactly one of:
      - "bilateral_weight_bearing_or_sitting"    (both feet planted evenly, walking, or seated)
      - "unilateral_feather_or_unstable"         (weight on one leg, tip-toe, unstable base, squatting on one leg)

    Check explicitly:
      knee_30_60_flexion:   are one or both knees bent between 30° and 60°?
      knee_gt_60_flexion:   are one or both knees bent more than 60°
                            (excluding sitting)? Deep squat / kneeling counts.

    ------------------------------------------------------------
    UPPER ARMS  (assess the more loaded / more extreme arm)
    ------------------------------------------------------------
    Observation: describe arm position RELATIVE TO THE TRUNK. First identify
    the trunk line (hip to shoulder); then read arm angle from that line.

    position_class must be exactly one of:
      - "ext_20_to_flex_20"                      (arm roughly aligned with trunk, ±20°)
      - "ext_gt_20_or_flex_20_to_45"             (arm extended behind trunk > 20°, OR flexed forward 20-45° from trunk)
      - "flex_45_to_90"                          (arm flexed 45-90° from trunk — forearm reaching forward/outward)
      - "flex_gt_90"                             (arm raised high, shoulder-height or above relative to trunk)

    CRITICAL: "flex_20_to_45" is NOT a valid value. If the arm is flexed
    20-45° from the trunk, the correct enum is "ext_gt_20_or_flex_20_to_45".

    Check explicitly:
      is_abducted_rotated: is the arm moved AWAY from the body sideways
        (elbow pointing outward), or rotated at the shoulder (palm facing
        unusual direction)?
      is_shoulder_raised: is the scapula / shoulder visibly lifted toward
        the ear (shrugged posture)?
      is_gravity_assisted: is the worker leaning on the arm, supporting
        its weight on an object, or is the arm hanging with gravity
        assisting (e.g. arm dangling while the trunk is bent forward)?

    ------------------------------------------------------------
    LOWER ARMS
    ------------------------------------------------------------
    Observation: describe the elbow angle (angle between upper and lower arm).

    position_class must be exactly one of:
      - "flex_60_to_100"                         (elbow in the mid-range, 60-100° — the neutral / comfortable range)
      - "flex_lt_60_or_gt_100"                   (elbow nearly straight < 60°, OR strongly bent > 100°)

    (Think of 60-100° as the natural resting elbow angle. Anything
    outside that range — whether the arm is nearly straight or the hand
    is brought very close to the shoulder — is awkward and gets the
    non-neutral enum.)

    ------------------------------------------------------------
    WRISTS
    ------------------------------------------------------------
    Observation: describe wrist posture (flat with forearm, bent up, bent
    down, deviated side to side).

    position_class must be exactly one of:
      - "flex_ext_0_to_15"                       (wrist roughly neutral, bent up or down less than 15°)
      - "flex_ext_gt_15"                         (wrist clearly bent up or down, more than 15°)

    Check softly (wrist deviation can be hard to see; if you are not
    sure, "false" is acceptable):
      is_deviated_twisted: is the hand bent toward the thumb or pinky
        side, or is the forearm twisted such that the wrist is rotated?

    ------------------------------------------------------------
    LOAD / FORCE
    ------------------------------------------------------------
    Observation: describe what the worker is holding or exerting force on,
    and your best estimate of its weight.

    category_class must be exactly one of:
      - "lt_5_kg"                                (empty hands, small tools, light box)
      - "5_to_10_kg"                             (medium container, one-handed tool use with resistance)
      - "gt_10_kg"                               (clearly heavy box, heavy bag, large load)

    Check explicitly:
      has_shock: is there a sudden impact or rapid force build-up
        (dropping, catching, jerking)?

    ------------------------------------------------------------
    COUPLING  (grip quality)
    ------------------------------------------------------------
    Observation: describe how the worker is holding the object or surface.

    category_class must be exactly one of:
      - "good_power_grip"                        (well-fitting handle, full power grip)
      - "fair_acceptable_hold"                   (acceptable hold but not ideal, or grip via another body part)
      - "poor_not_acceptable"                    (awkward hold, fingertip grip, slipping)
      - "unacceptable_unsafe"                    (no handles, awkward/unsafe grip, clearly dangerous)

    ============================================================
    OUTPUT FORMAT
    ============================================================

    Return ONLY a single JSON object matching the schema below. Do NOT
    include any text before or after the JSON, and do NOT use Markdown
    code fences.

    Every field is required. The observation field is free text (one
    sentence). All other string values must be chosen verbatim from the
    enum values listed above.

    {
      "group_a": {
        "trunk": {
          "observation": "<one sentence>",
          "position_class": "<one of the trunk enum values>",
          "is_twisted_sidebent": <true|false>
        },
        "neck": {
          "observation": "<one sentence>",
          "position_class": "<one of the neck enum values>",
          "is_twisted_sidebent": <true|false>
        },
        "legs": {
          "observation": "<one sentence>",
          "position_class": "<one of the legs enum values>",
          "knee_30_60_flexion": <true|false>,
          "knee_gt_60_flexion": <true|false>
        }
      },
      "group_b": {
        "upper_arms": {
          "observation": "<one sentence describing arm angle relative to trunk>",
          "position_class": "<one of the upper_arms enum values>",
          "is_abducted_rotated": <true|false>,
          "is_shoulder_raised": <true|false>,
          "is_gravity_assisted": <true|false>
        },
        "lower_arms": {
          "observation": "<one sentence>",
          "position_class": "<one of the lower_arms enum values>"
        },
        "wrists": {
          "observation": "<one sentence>",
          "position_class": "<one of the wrists enum values>",
          "is_deviated_twisted": <true|false>
        }
      },
      "context": {
        "load_force": {
          "observation": "<one sentence>",
          "category_class": "<one of the load_force enum values>",
          "has_shock": <true|false>
        },
        "coupling": {
          "observation": "<one sentence>",
          "category_class": "<one of the coupling enum values>"
        }
      }
    }
    """)


# Method trailers (same contract as v1)
METHOD_A_TRAILER = "\nAssess the posture shown in the image(s) above."
METHOD_B_TRAILER = (
    "\nAssess the posture in the video above. "
    "The key moment to evaluate is at t={timestamp:.2f} seconds."
)


def build_prompt(method: str, timestamp: float = None) -> str:
    """
    Build the full v2 user-turn text given the input method.

    Parameters
    ----------
    method : 'A' (images) or 'B' (video)
    timestamp : required for method B; the keyframe time in seconds.
    """
    if method == "A":
        return V2_DETAILED_TEXT + METHOD_A_TRAILER
    if method == "B":
        if timestamp is None:
            raise ValueError("Method B requires a timestamp (seconds).")
        return V2_DETAILED_TEXT + METHOD_B_TRAILER.format(timestamp=timestamp)
    raise ValueError(f"Unknown method: {method!r}. Use 'A' or 'B'.")


# =============================================================================
# Enum-sync check at import time
# =============================================================================
# Unlike v1 (which programmatically built enum lists), v2 bakes the enum values
# into the human-readable text above. Verify at import that every enum value
# in reba_tables.ENUMS appears literally in the prompt.

def _verify_enums_in_text():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from reba_tables import ENUMS as _ENUMS

    missing = []
    for _, values in _ENUMS.items():
        for v in values:
            # check as quoted string, which is how it appears in the prompt
            if f'"{v}"' not in V2_DETAILED_TEXT:
                missing.append(v)
    if missing:
        raise RuntimeError(
            "v2_detailed.py: the following enum values from reba_tables.ENUMS "
            f"are missing from the prompt text verbatim: {missing}"
        )


_verify_enums_in_text()


if __name__ == "__main__":
    print("METHOD A prompt length:", len(build_prompt("A")), "chars")
    print("METHOD B prompt length:", len(build_prompt("B", timestamp=3.47)), "chars")
    print()
    print("First 800 chars of Method A prompt:")
    print("=" * 72)
    print(build_prompt("A")[:800])
    print("...")
    print()
    print("Last 800 chars of Method B prompt:")
    print("=" * 72)
    print(build_prompt("B", timestamp=3.47)[-800:])
    print()
    print("✓ All enum values verified present in prompt text verbatim.")