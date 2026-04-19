"""
v3_multiagent/part_agents.py
----------------------------
Eight specialized agents, one per REBA body/context part. Each agent:
  - Sees the same image/video as the scene primer
  - Receives the scene primer's description as context
  - Focuses EXCLUSIVELY on its assigned part
  - Returns a small JSON with position_class (or category_class) +
    adjustments + confidence level

Key design choices:
  - One prompt template for structure, content per-part varies
  - Confidence only on the main categorical (position/category), not on
    each boolean — keeps output short
  - Part-specific viewing rules copied/adapted from v2.1 (same rules,
    but targeted to this single part only)
  - The output JSON is a PARTIAL annotation shape — the orchestrator
    merges the 8 partials into a full annotation

Schema per part (simplified):
  {
    "position_class": "<enum>",        OR  "category_class": "<enum>" for load/coupling
    "confidence": "high" | "medium" | "low",
    <adjustment_fields>: true|false,
    ...
  }
"""

from textwrap import dedent
from typing import Dict, Any


# =============================================================================
# Per-part configuration
# =============================================================================
# Each part has: display name, enum key field, enum values with descriptions,
# adjustment fields, and part-specific viewing guidance.

PART_CONFIGS: Dict[str, Dict[str, Any]] = {
    "trunk": {
        "enum_field": "position_class",
        "enum_values": [
            ('"upright"', "standing straight, less than ~5° lean"),
            ('"flex_0_to_20_or_ext_0_to_20"', "slight forward or backward lean, up to 20°"),
            ('"flex_20_to_60_or_ext_gt_20"', "forward lean 20-60°, OR any backward lean > 20°"),
            ('"flex_gt_60"', "deeply bent forward, more than 60°"),
        ],
        "adjustments": [
            ("is_twisted_sidebent",
             "Is the torso rotated (shoulders not parallel to hips) or leaning sideways?"),
        ],
        "viewing_guidance": dedent("""\
            Trunk angle is measured from the upright body position. Look at
            the spine line from hip to shoulder. The angle between this line
            and vertical is the trunk flexion (forward) or extension (backward).

            IMPORTANT: Workers in these videos are often bent past 60° at the
            waist. If the spine is nearly horizontal or the head is near waist
            height, choose "flex_gt_60". Do not pick middle categories for
            clearly extreme postures."""),
        "extreme_reminder": dedent("""\
            REMINDER: "flex_gt_60" (deep forward bend) is very common in lifting,
            reaching to the floor, and patient-handling tasks. Select it when
            the torso is clearly folded forward."""),
    },

    "neck": {
        "enum_field": "position_class",
        "enum_values": [
            ('"flex_0_to_20"', "head nearly upright to slightly down, up to 20°"),
            ('"flex_gt_20_or_ext"', "head clearly bent forward > 20°, OR tilted back"),
        ],
        "adjustments": [
            ("is_twisted_sidebent",
             "Is the head turned to one side, or tilted toward one shoulder?"),
        ],
        "viewing_guidance": dedent("""\
            Neck angle is the head's tilt relative to an upright neutral
            position. Look at the line from the shoulders to the top of the
            head."""),
    },

    "legs": {
        "enum_field": "position_class",
        "enum_values": [
            ('"bilateral_weight_bearing_or_sitting"',
             "both feet planted evenly, walking, or seated"),
            ('"unilateral_feather_or_unstable"',
             "weight on one leg, tip-toe, unstable base, one-legged squat"),
        ],
        "adjustments": [
            ("knee_30_60_flexion",
             "Are one or both knees bent between 30° and 60°?"),
            ("knee_gt_60_flexion",
             "Are one or both knees bent more than 60° (excluding sitting)?"),
        ],
        "viewing_guidance": dedent("""\
            Judge stance stability and knee flexion separately. "Unilateral"
            means weight is clearly on one leg or the base is unstable (e.g.
            mid-step, balancing, twisting). Check whether weight is actually
            evenly distributed — don't default to bilateral just because
            both feet are visible.

            Knee flexion: deep squat, kneeling, or deep lunge count as
            >60° flexion. Normal standing with slightly bent knees is
            usually <30°, neither adjustment applies."""),
    },

    "upper_arms": {
        "enum_field": "position_class",
        "enum_values": [
            ('"ext_20_to_flex_20"',
             "arm aligned with trunk ±20° — roughly resting along torso line"),
            ('"ext_gt_20_or_flex_20_to_45"',
             "arm extended behind trunk > 20°, OR flexed 20-45° from trunk"),
            ('"flex_45_to_90"',
             "arm flexed 45-90° from trunk — forearm reaching forward/outward"),
            ('"flex_gt_90"',
             "arm raised high — at or above shoulder height relative to trunk"),
        ],
        "adjustments": [
            ("is_abducted_rotated",
             "Is the arm moved away from the body sideways (elbow outward), "
             "or rotated at the shoulder?"),
            ("is_shoulder_raised",
             "Is the scapula / shoulder visibly lifted toward the ear (shrugged)?"),
            ("is_gravity_assisted",
             "Is the worker actively supporting arm weight on a surface? "
             "(Passive gravity hang does NOT count.)"),
        ],
        "viewing_guidance": dedent("""\
            CRITICAL: Upper arm angle is measured RELATIVE TO THE TRUNK,
            not relative to vertical or the ground.

            COUNTERINTUITIVE EXAMPLE: if the worker is bent 70° forward at
            the waist and the arm hangs straight down, the arm LOOKS vertical
            but is roughly aligned with the (bent) trunk — that is
            "ext_20_to_flex_20", not "flex_gt_90".

            OPPOSITE: if the worker stands upright and raises the arm
            straight out in front to shoulder height, the arm is ~90°
            relative to the trunk — that is "flex_45_to_90" (close to
            "flex_gt_90").

            First mentally locate the trunk line (hip to shoulder), then
            read arm angle from there. Use the scene overview's trunk
            orientation as a reference.

            Assess the more heavily loaded or more extreme arm."""),
        "extreme_reminder": dedent("""\
            REMINDER: "flex_20_to_45" is NOT valid. If arm is 20-45° from trunk,
            use "ext_gt_20_or_flex_20_to_45". And workers reaching up or
            overhead commonly fall into "flex_gt_90" — don't avoid it."""),
    },

    "lower_arms": {
        "enum_field": "position_class",
        "enum_values": [
            ('"flex_60_to_100"',
             "elbow in the mid-range, 60-100° — the neutral range"),
            ('"flex_lt_60_or_gt_100"',
             "elbow nearly straight (< 60°) OR strongly bent (> 100°)"),
        ],
        "adjustments": [],  # lower_arms has no adjustments
        "viewing_guidance": dedent("""\
            Lower arm angle is the elbow joint angle — the angle between
            upper arm and forearm.

            60-100° is the natural resting / comfortable range. Anything
            outside (nearly straight OR hand drawn close to shoulder) is
            awkward and gets "flex_lt_60_or_gt_100"."""),
    },

    "wrists": {
        "enum_field": "position_class",
        "enum_values": [
            ('"flex_ext_0_to_15"',
             "wrist roughly neutral, bent up or down less than 15°"),
            ('"flex_ext_gt_15"',
             "wrist clearly bent up or down, more than 15°"),
        ],
        "adjustments": [
            ("is_deviated_twisted",
             "Is the hand bent toward thumb or pinky side, or is the wrist "
             "rotated off its natural axis by forearm twist?"),
        ],
        "viewing_guidance": dedent("""\
            Wrist angle is the hand's deviation from the forearm line.

            Wrists bent more than 15° are very common in tool use, lifting,
            and any task requiring precise hand positioning. If there is
            ANY visible bend at the wrist joint (hand not in line with
            forearm), choose "flex_ext_gt_15".

            For the is_deviated_twisted adjustment: carrying an object
            whose shape forces the hand off-axis typically makes this
            true."""),
    },

    "load_force": {
        "enum_field": "category_class",
        "enum_values": [
            ('"lt_5_kg"', "empty hands, small tools, light box / bag"),
            ('"5_to_10_kg"', "medium container, one-handed tool use with resistance"),
            ('"gt_10_kg"', "heavy box, heavy bag, large load, heavy pushing"),
        ],
        "adjustments": [
            ("has_shock", "Is there sudden impact, drop, or rapid force build-up?"),
        ],
        "viewing_guidance": dedent("""\
            Estimate the weight of what the worker is holding or pushing
            against. Use visual cues: size of box, density of contents,
            how strenuously the worker is moving. A human patient being
            supported counts as > 10 kg."""),
    },

    "coupling": {
        "enum_field": "category_class",
        "enum_values": [
            ('"good_power_grip"', "well-fitting handle, full power grip"),
            ('"fair_acceptable_hold"',
             "acceptable hold but not ideal, OR grip via another body part"),
            ('"poor_not_acceptable"',
             "awkward hold, fingertip grip, slipping"),
            ('"unacceptable_unsafe"',
             "no handles, awkward/unsafe grip, clearly dangerous"),
        ],
        "adjustments": [],  # coupling has no adjustments
        "viewing_guidance": dedent("""\
            Assess grip quality. A tool with a designed handle + full
            grip = good. Supporting a patient with arms wrapped around
            torso = fair (grip via body part). Gripping a box at the
            edges with fingertips = poor. No clear grip at all = unacceptable."""),
    },
}


# =============================================================================
# Prompt assembly
# =============================================================================

def _format_enum_block(enum_values, enum_field):
    """Render the allowed values with descriptions."""
    lines = [f'  {enum} ' + ' ' * max(0, 44 - len(enum)) + f'({desc})'
             for enum, desc in enum_values]
    return "\n".join(lines)


def _format_adjustments_block(adjustments):
    if not adjustments:
        return "  (no adjustments for this part)"
    lines = []
    for field, desc in adjustments:
        lines.append(f'  "{field}": {desc}')
    return "\n".join(lines)


def _build_output_schema(part_name: str, cfg: Dict[str, Any]) -> str:
    """Construct the JSON output schema for this specific part."""
    lines = ['{']
    enum_field = cfg["enum_field"]
    lines.append(f'  "{enum_field}": "<one of the allowed values above>",')
    lines.append('  "confidence": "high" | "medium" | "low",')
    for field, _ in cfg["adjustments"]:
        lines.append(f'  "{field}": <true|false>,')
    # drop trailing comma from last line
    if lines[-1].endswith(','):
        lines[-1] = lines[-1][:-1]
    lines.append('}')
    return "\n".join(lines)


BASE_TEMPLATE = dedent("""\
    You are a specialized posture analyzer. Assess ONLY the {part_name_upper}
    of the worker. Ignore all other body parts.

    SCENE CONTEXT (from first-pass overview):
    {scene_context}

    === VIEWING GUIDANCE FOR {part_name_upper} ===
    {viewing_guidance}

    === ALLOWED VALUES ===
    The {enum_field} must be EXACTLY one of these (copy verbatim, do not
    shorten or modify):

    {enum_block}
    {extreme_reminder}

    === ADJUSTMENTS TO CHECK ===
    {adjustments_block}

    Answer each adjustment with true or false. Actively look for them
    — do not default to false without checking.

    === CONFIDENCE ===
    Also report your confidence in the {enum_field}:
      "high"   — visual evidence is clear, you are certain
      "medium" — reasonable inference, some uncertainty (angle, occlusion)
      "low"    — basic guess, insufficient visual evidence

    === OUTPUT ===
    Return ONLY a single JSON object matching this schema. No text
    before or after. No markdown fences.

    {output_schema}
    """)


def build_part_prompt(part_name: str, scene_description: str) -> str:
    """
    Build the agent prompt for a given part, incorporating the scene primer's
    description as context.
    """
    if part_name not in PART_CONFIGS:
        raise ValueError(f"Unknown part: {part_name}. "
                         f"Known: {list(PART_CONFIGS.keys())}")
    cfg = PART_CONFIGS[part_name]

    extreme_reminder = cfg.get("extreme_reminder", "").strip()
    if extreme_reminder:
        extreme_reminder_block = "\n" + extreme_reminder + "\n"
    else:
        extreme_reminder_block = ""

    return BASE_TEMPLATE.format(
        part_name_upper=part_name.upper().replace("_", " "),
        scene_context=scene_description.strip(),
        viewing_guidance=cfg["viewing_guidance"].strip(),
        enum_field=cfg["enum_field"],
        enum_block=_format_enum_block(cfg["enum_values"], cfg["enum_field"]),
        extreme_reminder=extreme_reminder_block,
        adjustments_block=_format_adjustments_block(cfg["adjustments"]),
        output_schema=_build_output_schema(part_name, cfg),
    )


# Method-specific trailers (identical to other prompt versions)
METHOD_A_TRAILER = "\nAssess the posture shown in the image(s) above."
METHOD_B_TRAILER = (
    "\nAssess the posture in the video above. "
    "The key moment is at t={timestamp:.2f} seconds."
)


def build_full_part_prompt(part_name: str,
                           scene_description: str,
                           method: str,
                           timestamp: float = None) -> str:
    """Complete prompt for a part agent, including method trailer."""
    core = build_part_prompt(part_name, scene_description)
    if method == "A":
        return core + METHOD_A_TRAILER
    if method == "B":
        if timestamp is None:
            raise ValueError("Method B requires a timestamp.")
        return core + METHOD_B_TRAILER.format(timestamp=timestamp)
    raise ValueError(f"Unknown method: {method!r}")


# =============================================================================
# Ordered list of parts (for orchestrator iteration)
# =============================================================================

PART_ORDER = [
    "trunk", "neck", "legs",
    "upper_arms", "lower_arms", "wrists",
    "load_force", "coupling",
]


# The (group, part) mapping for downstream assembly into full annotation shape
PART_TO_GROUP = {
    "trunk": "group_a",
    "neck": "group_a",
    "legs": "group_a",
    "upper_arms": "group_b",
    "lower_arms": "group_b",
    "wrists": "group_b",
    "load_force": "context",
    "coupling": "context",
}


# =============================================================================
# Enum-sync check at import time
# =============================================================================

def _verify_enums_synced():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from reba_tables import ENUMS as _ENUMS

    part_to_enum_key = {
        "trunk": "trunk.position_class",
        "neck": "neck.position_class",
        "legs": "legs.position_class",
        "upper_arms": "upper_arms.position_class",
        "lower_arms": "lower_arms.position_class",
        "wrists": "wrists.position_class",
        "load_force": "load_force.category_class",
        "coupling": "coupling.category_class",
    }
    for part, cfg in PART_CONFIGS.items():
        local = [v[0].strip('"') for v in cfg["enum_values"]]
        canonical = _ENUMS[part_to_enum_key[part]]
        if list(local) != list(canonical):
            raise RuntimeError(
                f"Part {part} enum out of sync with reba_tables.ENUMS:\n"
                f"  local:     {local}\n"
                f"  canonical: {canonical}"
            )


_verify_enums_synced()


if __name__ == "__main__":
    sample_scene = (
        "The worker is lifting a cardboard box from a low shelf to chest "
        "height, leaning forward approximately 45 degrees at the waist "
        "with both hands gripping the sides of the box."
    )
    print("Total parts:", len(PART_ORDER))
    print("\n" + "=" * 72)
    print("Sample: trunk agent prompt")
    print("=" * 72)
    print(build_full_part_prompt("trunk", sample_scene, "A"))
    print("\n" + "=" * 72)
    print("Sample: upper_arms agent prompt")
    print("=" * 72)
    print(build_full_part_prompt("upper_arms", sample_scene, "A"))
    print()
    print("✓ All enum lists synced with reba_tables.ENUMS.")

    # Print token-ish estimates
    print("\nPrompt char lengths (before + method trailer):")
    for part in PART_ORDER:
        p = build_full_part_prompt(part, sample_scene, "A")
        print(f"  {part:14s}: {len(p):>5d} chars")