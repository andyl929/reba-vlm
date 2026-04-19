"""
v2_1_detailed.py
----------------
Prompt v2.1: iterative refinement on v2 based on analysis of v2 results.

Core changes vs v2:
  - E6 REMOVED: the "if unsure, false is acceptable" guidance on wrist
    was contaminating the model's whole-body stance — in v2 the FN rate on
    wrists.is_deviated_twisted hit 100% (all 25 true cases missed). The
    replacement treats wrists like every other adjustment: hard, specific
    checking.
  - E5 RELOCATED: the anti-conservatism warning in v2 was at the top of
    the prompt; by the time the model reached each enum list, the
    reminder was gone from its working context. v2.1 places a specific
    reminder IMMEDIATELY BELOW each enum list that contains an extreme
    category (trunk, upper_arms, legs).
  - E3 STRENGTHENED: in v2 the model's observation field was frequently
    boilerplate ("the wrist is in a relatively neutral position")
    identical across different videos. v2.1 requires observations to
    contain a concrete angle estimate or a specific spatial relationship
    — boilerplate is explicitly forbidden.
  - E1 STRENGTHENED: upper arm angle relative to trunk — v2 stated the
    rule but gave no example. v2.1 gives the counterintuitive case:
    when a worker bends forward and the arm hangs by gravity, the arm
    may LOOK vertical to the camera yet still be ~0° relative to the
    (bent) trunk.

Unchanged from v2:
  - E2 anti-abbreviation enum enforcement (worked perfectly, 100% schema
    compliance in v2, preserved verbatim)
  - Output schema: identical to v2 (observation + enum fields)
"""

from textwrap import dedent


V2_1_DETAILED_TEXT = dedent("""\
    You are assessing a worker's posture using Rapid Entire Body Assessment
    (REBA), per Hignett & McAtamney (2000).

    You will score 8 body / context components. For each component, first
    describe what you see with SPECIFIC detail ("observation"), then assign
    the categorical class, then answer any yes/no adjustment questions.

    ============================================================
    IMPORTANT VIEWING PRINCIPLES
    ============================================================

    1. OBSERVATIONS MUST BE SPECIFIC
       Every observation field must contain a concrete visual statement:
       an approximate angle estimate (e.g. "~45° forward flexion"), a
       specific spatial relationship (e.g. "shoulders not aligned, left
       shoulder is clearly forward"), or a concrete action description
       (e.g. "hand is gripping a cylindrical tool with thumb over fingers").

       BOILERPLATE IS FORBIDDEN. Phrases like "in a relatively neutral
       position", "appears normal", "is held naturally", "is in a standard
       posture" are not acceptable observations — they indicate you did
       not look carefully. If a part truly looks neutral, write what
       that means specifically (e.g. "wrist is straight, aligned with
       the forearm, angle < 5°").

    2. ANGULAR REFERENCE FRAMES
       - TRUNK and NECK angles are measured from the UPRIGHT body position.
       - UPPER ARM angle is measured RELATIVE TO THE TRUNK, not relative
         to vertical or to the ground.

         COUNTERINTUITIVE EXAMPLE: if a worker is bent 70° forward at the
         waist and their arm hangs straight down from the shoulder, the
         arm may LOOK vertical from the camera. But relative to the
         (forward-bent) trunk, the arm is roughly aligned — that is
         "ext_20_to_flex_20", not "flex_gt_90".

         OPPOSITE CASE: if a worker stands upright and raises their arm
         straight out in front at shoulder height, the arm is roughly
         horizontal to the ground AND 90° relative to the trunk — that
         is "flex_45_to_90" (close to flex_gt_90).

         Always mentally align with the torso line first, then read arm
         angle from there.
       - LOWER ARM angle is the elbow joint angle.
       - WRIST angle is the hand's deviation from the forearm line.

    3. EXTREME POSTURES ARE COMMON IN THIS DATASET
       Many workers in these videos are in extreme postures: trunks bent
       past 60°, arms raised overhead, knees in deep flexion. Do NOT
       default to "moderate" middle-range categories when the posture is
       clearly extreme. Under-scoring by choosing safer middle categories
       was the #1 failure mode in prior experiments.

    4. ACTIVELY CHECK ADJUSTMENTS
       For every boolean adjustment, deliberately look for it. Do not
       answer "false" unless you have specifically looked and confirmed
       the deviation is absent. Visual deviations (twisted torso, raised
       shoulder, abducted arm, deviated wrist) are common in awkward
       work postures — missing them is the most common error mode.

    5. ASSESS THE MORE LOADED ARM
       If arms are in different positions, score the more heavily loaded
       or more extreme arm.

    ============================================================
    COMPONENT DEFINITIONS
    ============================================================

    The position_class and category_class strings below MUST be copied
    VERBATIM into your JSON output. Do not shorten them. Several enums
    contain BOTH a flexion and an extension clause joined by "_or_" —
    the full string is required.

    ------------------------------------------------------------
    TRUNK
    ------------------------------------------------------------
    Observation must include: approximate lean angle (forward or backward)
    in degrees, and whether the spine looks aligned or twisted/tilted.

    position_class must be exactly one of:
      - "upright"                                (standing straight, < ~5° lean)
      - "flex_0_to_20_or_ext_0_to_20"            (slight forward or backward lean, up to 20°)
      - "flex_20_to_60_or_ext_gt_20"             (forward lean 20-60°, OR any backward lean > 20°)
      - "flex_gt_60"                             (deeply bent forward, more than 60°)

    REMINDER: Many trunks in this dataset are bent past 60°. If the torso
    is clearly folded forward such that the head is near waist level, or
    the spine is nearly parallel to the ground, choose "flex_gt_60". Do
    not reflexively pick the middle category.

    Check explicitly:
      is_twisted_sidebent: is the torso rotated (one shoulder clearly
        further from the camera than the other, or shoulders not
        parallel to hips) or leaning to the side (spine not vertical
        relative to pelvis)?

    ------------------------------------------------------------
    NECK
    ------------------------------------------------------------
    Observation must include: approximate head tilt angle, and whether
    the head is facing forward, turned, or tilted to one side.

    position_class must be exactly one of:
      - "flex_0_to_20"                           (head nearly upright to slightly down, up to 20°)
      - "flex_gt_20_or_ext"                      (head clearly bent forward > 20°, OR tilted back)

    Check explicitly:
      is_twisted_sidebent: is the head turned to one side, or tilted
        toward one shoulder? This is VERY common in workers reaching
        sideways or attending to an object not directly in front.

    ------------------------------------------------------------
    LEGS
    ------------------------------------------------------------
    Observation must include: stance type (standing / walking / sitting
    / kneeling / squatting / one-legged), and approximate knee angle if
    knees are visibly bent.

    position_class must be exactly one of:
      - "bilateral_weight_bearing_or_sitting"    (both feet planted evenly, walking, or seated)
      - "unilateral_feather_or_unstable"         (weight on one leg, tip-toe, unstable base, one-legged squat)

    REMINDER: "unilateral_feather_or_unstable" is common when a worker
    is reaching, twisting, or balancing during the task. Do not default
    to "bilateral" just because both feet are visible — check whether
    weight is actually evenly distributed.

    Check explicitly:
      knee_30_60_flexion:  are one or both knees bent between 30° and 60°?
      knee_gt_60_flexion:  are one or both knees bent more than 60°
                           (excluding sitting)? Deep squat, kneeling,
                           or deep lunge all count.

    ------------------------------------------------------------
    UPPER ARMS  (assess the more loaded / more extreme arm)
    ------------------------------------------------------------
    Observation must include: the angle of the upper arm RELATIVE TO THE
    TRUNK LINE (not relative to vertical), and a description like "arm
    raised forward to shoulder height relative to bent torso" or
    "arm hanging aligned with torso".

    position_class must be exactly one of:
      - "ext_20_to_flex_20"                      (arm aligned with trunk ±20° — roughly resting along the torso line)
      - "ext_gt_20_or_flex_20_to_45"             (arm extended behind trunk > 20°, OR flexed 20-45° from trunk)
      - "flex_45_to_90"                          (arm flexed 45-90° from trunk — forearm reaching forward/outward)
      - "flex_gt_90"                             (arm raised high — at or above shoulder height relative to trunk)

    CRITICAL: "flex_20_to_45" is NOT a valid value. If the arm is flexed
    20-45° from the trunk, use "ext_gt_20_or_flex_20_to_45" instead.

    REMINDER: If a worker is reaching up, forward, or out to manipulate
    something at or above shoulder level, the correct enum is usually
    "flex_45_to_90" or "flex_gt_90" — not the middle categories. Workers
    performing overhead or extended-reach tasks commonly fall into
    "flex_gt_90".

    Check explicitly:
      is_abducted_rotated: is the arm moved AWAY from the body sideways
        (elbow pointing outward, not tucked in), or rotated at the
        shoulder (palm facing an unusual direction for the task)? This
        is very common when workers reach to the side or handle objects
        that require wrist rotation.
      is_shoulder_raised: is the scapula / shoulder visibly lifted toward
        the ear (shrugged posture)?
      is_gravity_assisted: is the worker leaning on the arm, supporting
        its weight on an object or surface? (Note: an arm hanging
        passively from gravity does NOT by itself make this true — it
        must be actively supported.)

    ------------------------------------------------------------
    LOWER ARMS
    ------------------------------------------------------------
    Observation must include: approximate elbow angle (the angle between
    the upper arm and the forearm).

    position_class must be exactly one of:
      - "flex_60_to_100"                         (elbow in the mid-range, 60-100° — the neutral / comfortable range)
      - "flex_lt_60_or_gt_100"                   (elbow nearly straight < 60°, OR strongly bent > 100°)

    REMINDER: If the worker's arm is nearly straight out (elbow barely
    bent) or the hand is drawn very close to the shoulder (elbow fully
    closed), the correct answer is "flex_lt_60_or_gt_100". Only
    mid-range elbow angles count as the neutral category.

    ------------------------------------------------------------
    WRISTS
    ------------------------------------------------------------
    Observation must include: specific description of wrist alignment
    with the forearm. State whether the wrist is straight, bent up
    (extension), bent down (flexion), deviated toward thumb, deviated
    toward pinky, or rotated with the forearm.

    position_class must be exactly one of:
      - "flex_ext_0_to_15"                       (wrist roughly neutral, bent up or down less than 15°)
      - "flex_ext_gt_15"                         (wrist clearly bent up or down, more than 15°)

    REMINDER: Wrists bent more than 15° from the forearm line are very
    common in tool use, lifting, and any task requiring precise hand
    positioning. If the hand is clearly not in line with the forearm
    (a visible bend at the wrist), choose "flex_ext_gt_15".

    Check explicitly:
      is_deviated_twisted: is the hand bent toward the thumb side or
        toward the pinky side (radial or ulnar deviation), or is the
        forearm twisted such that the wrist is rotated off its natural
        axis? When lifting or carrying an object whose shape forces
        the hand off-axis, this is typically true.

    ------------------------------------------------------------
    LOAD / FORCE
    ------------------------------------------------------------
    Observation must include: what the worker is holding or exerting
    force on, and your best weight estimate.

    category_class must be exactly one of:
      - "lt_5_kg"                                (empty hands, small tools, light box / bag)
      - "5_to_10_kg"                             (medium container, one-handed tool use with resistance)
      - "gt_10_kg"                               (clearly heavy box, heavy bag, large load, pushing heavy object)

    Check explicitly:
      has_shock: is there a sudden impact, drop, or rapid force build-up?

    ------------------------------------------------------------
    COUPLING  (grip quality)
    ------------------------------------------------------------
    Observation must include: specifically how the worker is holding the
    object or surface (full hand grip, fingertip grip, hugging with arm,
    palm flat against object, etc.).

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

    Every field is required. The observation field must be concrete and
    specific (not boilerplate). All other string values must be chosen
    verbatim from the enum values listed above.

    {
      "group_a": {
        "trunk": {
          "observation": "<specific description, include approximate angle>",
          "position_class": "<one of the trunk enum values>",
          "is_twisted_sidebent": <true|false>
        },
        "neck": {
          "observation": "<specific description, include approximate angle>",
          "position_class": "<one of the neck enum values>",
          "is_twisted_sidebent": <true|false>
        },
        "legs": {
          "observation": "<specific description of stance and knees>",
          "position_class": "<one of the legs enum values>",
          "knee_30_60_flexion": <true|false>,
          "knee_gt_60_flexion": <true|false>
        }
      },
      "group_b": {
        "upper_arms": {
          "observation": "<specific description of arm angle relative to trunk>",
          "position_class": "<one of the upper_arms enum values>",
          "is_abducted_rotated": <true|false>,
          "is_shoulder_raised": <true|false>,
          "is_gravity_assisted": <true|false>
        },
        "lower_arms": {
          "observation": "<specific description of elbow angle>",
          "position_class": "<one of the lower_arms enum values>"
        },
        "wrists": {
          "observation": "<specific description of wrist alignment with forearm>",
          "position_class": "<one of the wrists enum values>",
          "is_deviated_twisted": <true|false>
        }
      },
      "context": {
        "load_force": {
          "observation": "<what is being held, with weight estimate>",
          "category_class": "<one of the load_force enum values>",
          "has_shock": <true|false>
        },
        "coupling": {
          "observation": "<specific description of grip>",
          "category_class": "<one of the coupling enum values>"
        }
      }
    }
    """)


METHOD_A_TRAILER = "\nAssess the posture shown in the image(s) above."
METHOD_B_TRAILER = (
    "\nAssess the posture in the video above. "
    "The key moment to evaluate is at t={timestamp:.2f} seconds."
)


def build_prompt(method: str, timestamp: float = None) -> str:
    if method == "A":
        return V2_1_DETAILED_TEXT + METHOD_A_TRAILER
    if method == "B":
        if timestamp is None:
            raise ValueError("Method B requires a timestamp (seconds).")
        return V2_1_DETAILED_TEXT + METHOD_B_TRAILER.format(timestamp=timestamp)
    raise ValueError(f"Unknown method: {method!r}. Use 'A' or 'B'.")


# =============================================================================
# Enum-sync check
# =============================================================================

def _verify_enums_in_text():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from reba_tables import ENUMS as _ENUMS

    missing = []
    for _, values in _ENUMS.items():
        for v in values:
            if f'"{v}"' not in V2_1_DETAILED_TEXT:
                missing.append(v)
    if missing:
        raise RuntimeError(
            "v2_1_detailed.py: the following enum values from reba_tables.ENUMS "
            f"are missing from the prompt text verbatim: {missing}"
        )


_verify_enums_in_text()


if __name__ == "__main__":
    print("METHOD A prompt length:", len(build_prompt("A")), "chars")
    print("METHOD B prompt length:", len(build_prompt("B", timestamp=3.47)), "chars")
    print()
    print("✓ All enum values verified present in prompt text verbatim.")