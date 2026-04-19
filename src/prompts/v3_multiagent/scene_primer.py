"""
v3_multiagent/scene_primer.py
------------------------------
First-pass agent in the v3 multi-agent architecture.

Role: Produce a concise (2-4 sentence) scene description that captures the
worker's overall posture and task. This description is then fed as context
to the 8 downstream part-specific agents, giving each of them global
awareness without requiring them to re-analyze the full scene.

Design choices:
  - Output is PLAIN TEXT, not JSON (no enum overhead, no parsing risk)
  - Kept short to minimize downstream prompt bloat (part agents append
    this whole description to their own prompts)
  - Explicitly asks for angle estimates and load information — these are
    what part agents most need as context
  - No REBA terminology — just plain description. The part agents are
    the ones making REBA-specific judgments.

Takes either images (Method A) or a video (Method B); same prompt text
either way.
"""

from textwrap import dedent


SCENE_PRIMER_TEXT = dedent("""\
    You are the first stage of a two-stage posture analysis system.
    Your job is to provide a concise scene overview that will be used
    as context by downstream analyzers.

    Describe what you see in 3-4 short sentences. Include:

    1. What the worker is doing (the task / activity).
    2. Overall body orientation: is the trunk upright, leaning forward,
       leaning back, twisted? Roughly how many degrees?
    3. What the worker is holding or manipulating, and your rough
       estimate of its weight.
    4. Any notable posture features: raised arms, deep knee bend,
       one-legged stance, kneeling, etc.

    Be specific and concrete. Include approximate angles where possible
    (e.g. "bent forward roughly 60 degrees at the waist"). Avoid vague
    phrases like "relatively normal" or "typical posture".

    If multiple people are visible, describe ONLY the worker performing
    the primary task (the person actively handling objects or tools,
    not a passive recipient).

    Output just the description as plain text. Do NOT use JSON, bullet
    points, or markdown formatting.
    """)


METHOD_A_TRAILER = "\nDescribe the scene shown in the image(s) above."
METHOD_B_TRAILER = (
    "\nDescribe the scene in the video above. "
    "The key moment to focus on is at t={timestamp:.2f} seconds."
)


def build_scene_primer_prompt(method: str, timestamp: float = None) -> str:
    if method == "A":
        return SCENE_PRIMER_TEXT + METHOD_A_TRAILER
    if method == "B":
        if timestamp is None:
            raise ValueError("Method B requires a timestamp (seconds).")
        return SCENE_PRIMER_TEXT + METHOD_B_TRAILER.format(timestamp=timestamp)
    raise ValueError(f"Unknown method: {method!r}. Use 'A' or 'B'.")


if __name__ == "__main__":
    print("=" * 72)
    print("Method A scene primer:")
    print("=" * 72)
    print(build_scene_primer_prompt("A"))
    print()
    print("=" * 72)
    print("Method B scene primer:")
    print("=" * 72)
    print(build_scene_primer_prompt("B", timestamp=3.47))