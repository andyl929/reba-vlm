# REBA Video Analysis Project — Complete Walkthrough

**Author**: Andy Li (ali35), NCSU ISE
**Date**: April 2026
**Project duration**: 4 days of active development

---

## Preamble: What this document is

This walkthrough is a complete narrative of the project — not only the final results, but also the reasoning process, wrong turns, and design decisions that shaped the final approach. It is more detailed than the conference paper, intended as:

1. A companion to the paper's methodology section
2. A record of the iterative prompt engineering process (for reproducibility and teaching value)
3. A reference for future extensions of the work

The core experiments use **Gemma 4 31B** as a local multimodal LLM hosted on NCSU VCL. A second phase applying the same prompts to **Gemini 3.1 Pro** is planned.

---

## 1. Problem Statement and Motivation

### What is REBA?

Rapid Entire Body Assessment (REBA) is a widely-used ergonomic risk assessment tool from Hignett & McAtamney (2000). Given a worker's posture during a task, REBA produces a numerical score (1–15) classifying injury risk. The methodology decomposes the body into:

- **Group A**: trunk, neck, legs (scored from posture angle and adjustments)
- **Group B**: upper arms, lower arms, wrists
- **Context**: load weight, coupling (grip quality)

Sub-scores combine via three lookup tables (Table A, Table B, Table C) into a final risk score and action level (None / Low / Medium / High / Very High).

REBA is widely used in occupational health but is **labor-intensive**: a trained observer must assess each video frame-by-frame. In manufacturing, healthcare, and construction, this limits REBA deployment to small-scale studies.

### Research Question

**Can a multimodal LLM perform REBA assessment directly from video or keyframe input, given appropriate prompt engineering?**

Sub-questions:
- How do prompt design choices affect accuracy?
- Do single-agent or multi-agent architectures work better?
- What parts of REBA are tractable for LLMs, and what requires dedicated vision models?

### Novelty

Prior work on LLM-based posture analysis has largely focused on:
- Single-axis tasks (e.g. just trunk angle)
- Using LLMs as post-processors on top of pose estimation pipelines
- Free-text descriptions without structured output

Our contribution is **end-to-end REBA with structured enum outputs that plug directly into the standard lookup tables**. This is the first study (to our knowledge) applying a local multimodal LLM to the full REBA scoring pipeline.

---

## 2. Dataset

### Source
- **20 MP4 videos** collected from public sources (workplace ergonomics training videos, publicly posted manual-labor footage)
- Each video: 20-60 seconds, variable resolution (480p-1080p), mostly single-worker
- 2/20 videos contain a second person (REBA_10: nurse + patient; REBA_13: worker + helper)

### Annotation Process

A custom Streamlit annotation tool (`reba_app.py`) was built to:
1. Play video, extract frame at selected timestamp
2. Display radio buttons and checkboxes for every REBA field
3. Compute sub-scores, Table A/B/C values, and final REBA score in real time
4. Export as JSON with the full state

**Annotator**: Andy Li (ISE grad student, familiar with REBA methodology).

**Output**: 44 annotation JSONs, each representing one keyframe evaluation. Videos have 1-3 keyframes each:
- 2 videos × 1 keyframe
- 12 videos × 2 keyframes
- 6 videos × 3 keyframes

Multiple keyframes per video capture different task phases (e.g. lifting: pick-up, carry, place-down).

### Schema

Each annotation JSON follows the REBA structure:

```json
{
  "meta_data": {
    "video_file": "REBA_1.mp4",
    "timestamp_sec": 2.00,
    "annotator": "Andy Li",
    "annotation_time": "2026-04-10 00:19:24"
  },
  "group_a": {
    "trunk": {"position_class": "upright", "is_twisted_sidebent": false, "sub_score": 1},
    "neck": {"position_class": "flex_0_to_20", "is_twisted_sidebent": false, "sub_score": 1},
    "legs": {"position_class": "unilateral_feather_or_unstable", "knee_30_60_flexion": false, "knee_gt_60_flexion": false, "sub_score": 2}
  },
  "group_b": {
    "upper_arms": {"position_class": "flex_gt_90", "is_abducted_rotated": true, "is_shoulder_raised": true, "is_gravity_assisted": false, "sub_score": 6},
    "lower_arms": {"position_class": "flex_60_to_100", "sub_score": 1},
    "wrists": {"position_class": "flex_ext_0_to_15", "is_deviated_twisted": false, "sub_score": 1}
  },
  "context": {
    "load_force": {"category_class": "lt_5_kg", "has_shock": false, "sub_score": 0},
    "coupling": {"category_class": "fair_acceptable_hold", "sub_score": 1}
  },
  "scores": {
    "table_a": 2,
    "score_a": 2,
    "table_b": 7,
    "score_b": 8,
    "final_reba_score": 6,
    "action_level": {"level": 2, "risk": "Medium", "action_required": "Necessary"}
  }
}
```

### Data Quality Assurance

Two validation scripts verify the integrity of the 44 annotations:

1. **`validate_annotations.py`**: For every annotation, recompute each `sub_score` from `position_class + boolean adjustments` according to the REBA paper. Expected: 352/352 sub-scores (8 parts × 44 samples) match.

   Initial run found 53 mismatches — all due to Python enum naming differences between my early implementation and the Streamlit tool's labels (e.g. I wrote `flex_0_to_20` but the tool used `flex_0_to_20_or_ext_0_to_20`). After aligning names, **all 352 sub-scores match**.

   One additional anomaly: in REBA_9_2.49s.json, `coupling.sub_score=0` but `category_class="fair_acceptable_hold"` (which should be 1). This was a Streamlit rerun snapshot bug (the tool's display updated the sub_score asynchronously from the radio button selection). Fixed via a one-line patch script.

2. **`validate_full_pipeline.py`**: For every annotation, recompute all 5 scalar scores (`table_a`, `score_a`, `table_b`, `score_b`, `final_reba_score`) using my REBA implementation. Expected: all 44 annotations match their labeled scores.

   **Result: 44/44 perfect match**, confirming both (a) the ground truth is internally consistent, and (b) my implementation matches the original Streamlit tool's tables.

3. **Paper worked example**: Implemented the worked example from Hignett & McAtamney (2000, Fig. 2, a physiotherapist treating a stroke patient with final REBA=11). All intermediate sub-scores and lookup table values match.

---

## 3. Architecture & Infrastructure

### Hardware
- 2× NVIDIA A100-PCIE-40GB on NCSU VCL
- Ubuntu 24.04, CUDA 12.8, driver 570.211.01

### Local Model
- **Gemma 4 31B Instruct** (google/gemma-4-31B-it, 30.7B params, BF16)
- Released 2026-04-02, Apache 2.0 license
- 256K context capable (we limit to 8192 tokens for our use case)
- Multimodal: text + image + video, vision encoder fused into main safetensors (no separate mmproj)

### Inference Server
- **vLLM 0.19.0** serving OpenAI-compatible API on `localhost:8000`
- `--tensor-parallel-size 2` (splits model across both GPUs)
- `--gpu-memory-utilization 0.90`
- `--max-model-len 8192`
- `--limit-mm-per-prompt '{"image": 30, "video": 1}'`

### Video Processing

- **Method A (keyframe)**: `ffmpeg -ss {timestamp} -i {video} -frames:v 1 -q:v 2 {output.jpg}`. Uses accurate seek (slower but frame-precise). Extracted frames cached under `data/frames/`.
- **Method B (video)**: Full MP4 sent as base64 data URL. vLLM's Gemma 4 video handler extracts up to 32 frames at 70 tokens each, auto-inserting mm:ss timestamps between frames. The prompt hints "the key moment is at t=X.XXs" so the model can correlate.

---

## 4. REBA Pipeline Implementation

### Design Decision (D1): Separation of Concerns

**The LLM only predicts** `position_class` and `boolean adjustments`. All arithmetic — sub-score calculation, Table A/B/C lookups, final score — is handled by Python code.

Why?

1. REBA arithmetic is **definitional**: the Table A at `[trunk=2][neck=1][legs=3]` is defined as `3`, full stop. Asking an LLM to "compute REBA" obscures whether errors come from visual understanding or from the model learning table values incorrectly.
2. The **only interesting signal** is the model's categorical judgment (position_class, booleans). Our experiments are therefore purely about visual understanding, isolated from arithmetic.
3. If the model predicts position_class correctly, our code guarantees a correct final score. This makes error analysis clean.

### Implementation (`reba_tables.py`)

```python
# Sub-score computation (one function per part)
def compute_trunk(p):
    base = TRUNK_BASE.get(p["position_class"])      # 1, 2, 3, or 4
    if p.get("is_twisted_sidebent"):
        base += 1
    return base

# Table A lookup (from the annotation tool, verified against paper)
TABLE_A = [
    [[1,2,3,4],[1,2,3,4],[3,3,5,6]],
    [[2,3,4,5],[3,4,5,6],[4,5,6,7]],
    ...
]

def lookup_table_a(trunk, neck, legs):
    return TABLE_A[trunk-1][neck-1][legs-1]

# Full pipeline
def compute_full_reba(annotation, activity_score=0):
    sub_scores = {part: fn(annotation[group][part])
                  for (group,part), fn in COMPUTERS.items()}
    table_a = lookup_table_a(sub_scores["trunk"], sub_scores["neck"], sub_scores["legs"])
    score_a = table_a + sub_scores["load_force"]
    table_b = lookup_table_b(sub_scores["upper_arms"], sub_scores["lower_arms"], sub_scores["wrists"])
    score_b = table_b + sub_scores["coupling"]
    score_c = lookup_table_c(score_a, score_b)
    final = clamp(score_c + activity_score, 1, 15)
    return {...}
```

### Activity Score

**Hardcoded to 0** in all experiments.

The REBA Activity Score adds +1 for each of:
- Static posture >1 minute
- Repetition >4 times/minute
- Rapid large-range changes

These cannot be reliably determined from short keyframes (Method A) or short videos without domain context. They are also **not present in our ground-truth annotations** — the annotator focused on posture fields.

This is documented as a limitation. A future extension could predict activity from full-length video with longer context.

---

## 5. Experimental Pipeline

### Top-level Architecture

```
┌─────────────────┐
│ Video + GT JSON │
│ (44 annotations) │
└─────┬───────────┘
      │
      ├── Method A ── ffmpeg keyframe extraction ─┐
      │                                           │
      ├── Method B ── base64 video ───────────────┤
      │                                           │
      │                                           ▼
      │                                    ┌──────────────┐
      │                                    │  Prompt v1   │
      │                                    │  /v2/v2.1/v3 │
      │                                    └──────┬───────┘
      │                                           │
      │                                           ▼
      │                                    ┌──────────────┐
      │                                    │ vLLM (Gemma4) │
      │                                    └──────┬───────┘
      │                                           │
      │                                           ▼
      │                                    ┌──────────────┐
      │                                    │ JSON parse + │
      │                                    │  validate    │
      │                                    └──────┬───────┘
      │                                           │
      ▼                                           ▼
 ┌─────────────────────────────────────────────────────┐
 │  compute_full_reba(prediction)  →  score comparison │
 └─────────────────────────────────────────────────────┘
      │
      ▼
 JSONL record with ground_truth + prediction + computed_scores
```

### Reproducibility Settings

- `temperature=0.0` (greedy decoding)
- `seed=42` (fixed)
- Result: every experiment is fully deterministic. Re-running yields identical outputs byte-for-byte.

---

## 6. Prompt Version 1 — Baseline

### Design Philosophy

The v1 prompt translates the REBA worksheet (Hignett & McAtamney 2000) directly into a prompt, with minimum engineering. It is intentionally weak — a floor against which subsequent versions are compared.

### Key Design Decisions

- **Enum enforcement**: Include explicit lists of allowed enum values (e.g. `"trunk.position_class": one of ["upright", "flex_0_to_20_or_ext_0_to_20", ...]`). The alternative, free-text output, would make it impossible to compare predictions to ground truth labels.
- **JSON schema**: The prompt explicitly asks for a single JSON object with no surrounding text, no Markdown code fences. Without this, models often wrap output in ```json ... ``` blocks or add prose preambles.
- **No observation step**: The model is asked directly for classifications. No "first describe, then classify" scaffolding.
- **No worked examples**: No one-shot or few-shot examples. Pure zero-shot.
- **No industry context**: The prompt does not say "this is a construction worker" or similar. REBA is designed as a universal tool; injecting context would leak information.

### v1 Prompt Structure (3.7K characters)

```
You are assessing a worker's posture using REBA...

=== Group A definitions ===
Trunk position:
  - Upright
  - 0°-20° flexion, or 0°-20° extension
  - 20°-60° flexion, or >20° extension
  - >60° flexion
Adjustment: trunk is twisted or side-flexed (yes/no).

[... similar blocks for neck, legs, upper_arms, lower_arms, wrists, load_force, coupling ...]

OUTPUT:
Return ONLY a single JSON object matching:
{
  "group_a": {
    "trunk": {
      "position_class": one of ["upright", "flex_0_to_20_or_ext_0_to_20",
                                "flex_20_to_60_or_ext_gt_20", "flex_gt_60"],
      "is_twisted_sidebent": <true|false>
    },
    ...
  },
  "group_b": {...},
  "context": {...}
}
```

### v1 Results

| Metric | Method A | Method B |
|---|---|---|
| JSON parse rate | 100% | 100% |
| Schema compliance (ok) | **36.4%** | **45.5%** |
| Schema invalid | 63.6% | 54.5% |
| All-fields accuracy | 60.3% | 59.5% |
| Final REBA MAE (ok only) | 2.62 | 2.40 |
| Pearson r (ok only) | 0.11 | 0.25 |

### The Big Finding

**All 28 schema-invalid cases in Method A (and all 24 in Method B) had the exact same cause**: the model output `"flex_20_to_45"` for `upper_arms.position_class`. The correct enum is `"ext_gt_20_or_flex_20_to_45"`.

The REBA worksheet's second upper-arm category is "20° extension OR 20-45° flexion", so the enum name combines both clauses with `_or_`. When the worker is flexed (not extended), the model mentally "simplifies" the compound enum to just the flexion clause, producing the invalid abbreviation.

This is a clean, reproducible failure mode of Gemma 4 31B's enum adherence. It led directly to v2's E2 intervention.

### v1 Error Analysis Summary

**Pattern 1: Systematic under-estimation of extreme postures.**
- 15 samples had GT=flex_gt_60 (trunk). Only 2 matched; 11 were predicted as flex_20_to_60.
- Model prefers "moderate" categories even on visually clear extreme postures.

**Pattern 2: Systematic under-reporting of adjustments.**
- trunk.is_twisted_sidebent: 75% false-negative rate
- neck.is_twisted_sidebent: 92% false-negative rate
- upper_arms.is_abducted_rotated: 83% false-negative rate
- Model defaults booleans to `false`.

**Pattern 3: Poor handling of fine-grained fields.**
- upper_arms.position_class: 13.6% accuracy (Method A)
- Even ignoring schema_invalid issues, model can't reliably estimate arm angle.

---

## 7. Prompt Version 2 — Detailed

### Design Philosophy

v2 adds five explicit improvements based on v1 error analysis:

- **E1: Angular reference frames**. Upper arm angle is measured relative to trunk, not to vertical. This is the single biggest correction — v1 failed on bent-worker + arm-hanging cases because the model judged the arm's angle against the camera's vertical, not against the bent trunk.
- **E2: Anti-enum-abbreviation**. Explicit warning that `flex_20_to_45` is NOT valid; `ext_gt_20_or_flex_20_to_45` must be copied verbatim.
- **E3: Observation field**. Each part gets an `observation` field before its enum, encouraging visual-to-text reasoning.
- **E4: Active adjustment checking**. "Do not default booleans to false; deliberately look for each adjustment."
- **E5: Counter-regression-to-mean warning** at the top of the prompt.
- **E6: Wrist soft-guidance**. "If not sure about wrist deviation, false is acceptable." — later turned out to be a bug, see v2.1.

### v2 Prompt Structure (10.6K characters)

```
You are assessing a worker's posture using REBA...

=== IMPORTANT VIEWING PRINCIPLES ===
1. ANGULAR REFERENCE FRAMES
   - UPPER ARM angle is measured RELATIVE TO THE TRUNK, not vertical.
2. EXTREME POSTURES ARE COMMON
   Do NOT default to "moderate" categories...
3. ACTIVELY LOOK FOR ADJUSTMENTS
   ...

=== COMPONENT DEFINITIONS WITH EXACT ENUM VALUES ===
TRUNK
Observation: describe the torso's lean direction and roughly how many degrees from upright.
position_class must be exactly one of: [...]
Check explicitly: is_twisted_sidebent: ...

[... 8 parts, each with observation instruction, exact enum list, adjustments ...]

=== OUTPUT FORMAT ===
{
  "group_a": {
    "trunk": {
      "observation": "<one sentence>",
      "position_class": "<one of ...>",
      "is_twisted_sidebent": <true|false>
    },
    ...
  },
  ...
}
```

### v2 Results

| Metric | Method A | Method B |
|---|---|---|
| Schema compliance | **100%** | **100%** |
| All-fields accuracy | 62.7% | 59.9% |
| upper_arms.position_class | 34.1% (from 13.6%) | 27.3% (from 22.7%) |

### v2 Hits and Misses

**Hits:**
- **E2 was devastatingly effective**: schema invalid went to 0% on both methods.
- **E1 had real effect**: upper_arms accuracy jumped 20pp on Method A.

**Misses:**
- **E4 did NOT work on adjustments**: trunk.is_twisted_sidebent FN rate stayed at 100% on Method A. Model continued defaulting booleans to false.
- **E6 was actively harmful**: wrist.is_deviated_twisted hit **100% FN rate** (all 25 true-cases missed). The soft-guidance "if unsure, false is OK" was interpreted by the model as a universal license to hedge.
- **Observation field was boilerplate**: 40/44 wrist observations were literally "The wrists are held in a relatively neutral position" — same string repeated. The model was producing plausible-sounding descriptions without actually looking.

### The critical v2 insight

Looking at the raw observation field revealed something important:

```
[REBA_REBA_10_1.70s.json]  "observation": "The wrists are held in a relatively neutral position."
[REBA_REBA_10_7.50s.json]  "observation": "The wrists are held in a relatively neutral position."
[REBA_REBA_12_1.30s.json]  "observation": "The wrist is held relatively flat in line with the forearm while using the tool."
```

**Identical observations across completely different samples.** This is not a model "looking at the image" — this is a model generating the kind of sentence that "sounds right" for a wrist description.

This insight drove v2.1's main innovation.

---

## 8. Prompt Version 2.1 — Refined

### Design Philosophy

v2.1 makes three targeted fixes:

1. **Remove E6 soft-guidance** (bug): the wrist "if unsure, false" instruction contaminated boolean judgments across all parts. Replaced with hard instruction matching all other booleans.
2. **Relocate E5 reminders**: the "don't avoid extreme categories" warning was at the top of the prompt. By the time the model reached each enum list, the warning was out of its working attention. v2.1 places specific reminders **immediately below** each enum list that contains an extreme category (trunk, upper_arms, legs).
3. **Strengthen E3 observation**: explicit prohibition of boilerplate ("relatively", "appears", "typical"). Required concrete angle estimates or specific spatial relationships.

### v2.1 Prompt Excerpt (13.7K characters)

```
1. OBSERVATIONS MUST BE SPECIFIC
   Every observation field must contain a concrete visual statement:
   an approximate angle estimate (e.g. "~45° forward flexion"),
   a specific spatial relationship ("shoulders not aligned, left shoulder forward"),
   or a concrete action description.

   BOILERPLATE IS FORBIDDEN. Phrases like "in a relatively neutral position",
   "appears normal", "is held naturally" are not acceptable observations.

[... E1, E3, E4 carried over from v2 ...]

TRUNK
Observation must include: approximate lean angle in degrees.
position_class: [...]

REMINDER: Many trunks in this dataset are bent past 60°. If the torso
is clearly folded forward such that the head is near waist level, or
the spine is nearly parallel to the ground, choose "flex_gt_60". Do
not reflexively pick the middle category.

[... similar reminders for upper_arms, legs ...]
```

### v2.1 Results

| Metric | Method A | Method B |
|---|---|---|
| Schema compliance | 100% | 100% |
| All-fields accuracy | **65.9%** (+5.6 from v1, +3.2 from v2) | **63.9%** (+4.4 from v1) |
| Categorical mean | **56.2%** | **54.0%** |
| Boolean mean | **74.5%** | **72.7%** |
| wrists.position_class | 68.2% (from v2's 31.8%) | 70.5% (from 38.6%) |
| Final REBA Pearson r | **0.29** | 0.03 |

### Observation quality transformation

Wrist observations (Method A, v2 vs v2.1):

| Metric | v2 | v2.1 |
|---|---|---|
| Unique strings | 37/44 (84%) | 43/44 (98%) |
| Contains angle estimate | 2/44 (5%) | 44/44 (100%) |
| Contains boilerplate phrase | 40/44 | 20/44 |

Example of v2.1 observation (compare to the v2 boilerplate above):

```
REBA_REBA_10_1.70s: "The right wrist is slightly extended, approximately 10°
                     from the forearm line, as it grips the patient's side."

REBA_REBA_15_11.51s: "The wrist is bent slightly downward (flexion) by
                      approximately 20° to align the hammer with the nail."
```

Models now produce non-repeating, angle-specific descriptions.

### The Method B paradox

A counterintuitive v2.1 finding: on Method B, Pearson r for final_reba_score dropped from v1's 0.25 to v2.1's 0.03 — looking like a regression.

Explanation: in v1, only 20/44 samples produced valid schemas on Method B. Those 20 samples were a biased subset — the "easy" samples where the model could align to the strict format. Their correlation (0.25) over-represented the model's true performance. In v2.1, all 44 samples pass schema validation, and the true baseline correlation on video input is revealed to be lower than v1's sample-biased estimate.

This is a **methodology finding**: evaluation on the "ok" subset is systematically inflated. Honest comparison requires the same sample base across versions, which is only possible once schema compliance is 100%.

---

## 9. Prompt Version 3 — Multi-Agent

### Motivation

v2.1 hit two walls:
1. **Single-prompt attention budget**: v2.1's prompt is 3200 tokens. Adding more instructions yielded diminishing returns. The model can't attend equally to all 8 body parts simultaneously.
2. **Part-specific nuance**: some parts (load, coupling) need scene-level reasoning; others (wrist) need fine visual detail. A single prompt balances these concerns inefficiently.

Multi-agent decomposition aims to give each part its own focused attention.

### Architecture

```
    ┌─────────────────────────────┐
    │      Input: image/video      │
    └──────────────┬──────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Scene Primer   │    ◀── Agent 1
          │ (plain text)   │        Produces: 3-4 sentence
          └────────┬───────┘        scene description
                   │
            scene description
                   │
                   ▼
      ┌────────────────────────────────┐
      │  Parallel dispatch via          │
      │  ThreadPoolExecutor + vLLM batch│
      └─┬─┬─┬─┬─┬─┬─┬─┬───────────────┘
        │ │ │ │ │ │ │ │
        ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
     [trunk]  [neck]  [legs]        ◀── Agents 2-9
     [UA]    [LA]    [wrists]           Each sees: image + scene desc
     [load]  [coupling]                 Each outputs: partial JSON
        │ │ │ │ │ │ │ │
        ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
    ┌──────────────────────┐
    │   Python Aggregator  │        ◀── Not an agent
    │   merge + validate + │             Deterministic code
    │   compute_full_reba  │
    └──────────────────────┘
                   │
                   ▼
             Final prediction
```

### Scene Primer

Plain-text output (no JSON). 3-4 sentences describing:
1. What the worker is doing
2. Overall body orientation with approximate angles
3. What they're holding/manipulating (with weight estimate)
4. Any notable features (raised arms, deep knee bend, one-legged stance)

Explicit instruction to focus on a single worker in multi-person scenes.

### Part Agents

Each part agent receives:
- The same image/video input
- The scene primer's description as context
- A part-specific prompt with viewing rules, enum list, and adjustment checks
- Instruction to output a small JSON: `{enum_field, confidence: high|medium|low, <adjustments>}`

The part prompt is 1600-3000 characters (vs v2.1's 13700) — much more focused.

### Design Decisions for v3

| Decision | Choice | Rationale |
|---|---|---|
| Scene primer format | Plain text paragraph | Avoids JSON overhead; easier for model |
| Part agent input | Full image + scene description | No bounding box cropping (would require extra CV) |
| Cross-agent reasoning | None (fully parallel) | Preserves async benefit; adding dependencies forces serial |
| Higher-level agents (validator, critic) | None | LLM critics on same vision don't add info; code validator is sufficient |
| Confidence level | Main categorical only | Keep output compact; categorical captures most uncertainty |
| Parallelization | ThreadPoolExecutor (sync requests) | Avoids httpx async dependency; vLLM batches at engine level |

### v3 Results

| Metric | Method A | Method B |
|---|---|---|
| Schema compliance | 100% | 100% |
| All-fields accuracy | 63.5% | 62.6% |
| Final REBA MAE | 2.70 | **2.43** (best across all versions) |
| Mean bias | -0.52 | **-0.30** (lowest across all versions) |
| Pearson r | 0.12 | **0.27** (best on Method B) |
| Latency per sample | ~25 seconds | ~35 seconds |

### The v3 Paradox

**v3 field-level accuracy is lower than v2.1 (-2.4pp Method A, -1.3pp Method B).**

**But v3's final REBA score has the lowest bias and best correlation on Method B.**

This is not a contradiction. Analysis of per-field results shows the errors shift:
- v2.1 wrists Method B: 32/44 samples have GT=gt_15, model predicts 0_to_15 (under-estimation)
- v3 wrists Method B: 31/32 samples with GT=gt_15 correctly classified; **but 12/12 samples with GT=0_to_15 incorrectly predicted as gt_15** (over-estimation)

v3 flips the error direction. Because REBA final score is a sum of opposing errors across 8 parts, opposing errors partially cancel, reducing final-score bias even as field-level accuracy stays flat.

### Confidence Elicitation

Across all 352 part judgments (44 × 8, Method A), the model reported:
- "high" confidence: 283 judgments (80%)
- "medium" confidence: 69 judgments (20%)
- "low" confidence: **0 judgments**

Pooled accuracy by confidence:
- high: 53.4%
- medium: 49.3%
- **Gap: +4.1pp** (weak signal)

On Method B, the gap reverses to -1.8pp. The model's absolute confidence is poorly calibrated.

**But**: confidence patterns are informative **at the part level**. `load_force` and `coupling` show strong signals (21-43pp gaps). `wrists` is the only part with high medium-confidence rates (68% Method A, 48% Method B) — matching human annotator reports of wrist being the most subjective field.

**Finding**: model confidence is an unreliable uncertainty estimate in isolation, but the pattern of "where the model hedges" aligns with human assessments of task difficulty.

---

## 10. Cross-Version Comparison

### Overall Performance Table

| Metric | v1 M-A | v2 M-A | **v2.1 M-A** | v3 M-A | v1 M-B | v2 M-B | **v2.1 M-B** | v3 M-B |
|---|---|---|---|---|---|---|---|---|
| Schema compliance | 36% | 100% | 100% | 100% | 45% | 100% | 100% | 100% |
| All-fields accuracy | 60.3% | 62.7% | **65.9%** | 63.5% | 59.5% | 59.9% | **63.9%** | 62.6% |
| Categorical mean | 49.4% | 51.1% | **56.2%** | 52.6% | 48.0% | 46.6% | **54.0%** | 52.3% |
| Boolean mean | 69.9% | 73.0% | **74.5%** | 73.2% | 69.7% | 71.7% | **72.7%** | 71.7% |
| Final MAE | 2.62 | 2.64 | 2.64 | 2.70 | 2.40 | 2.64 | 2.77 | **2.43** |
| Final bias | -0.88 | -1.41 | -1.09 | -0.52 | -0.50 | -1.41 | -0.77 | **-0.30** |
| Pearson r | 0.11 | 0.26 | **0.29** | 0.12 | 0.25 | 0.21 | 0.03 | **0.27** |

### Key Takeaways

1. **Schema compliance is "solved"** by v2+. The jump from 36-45% to 100% shows the power of explicit enum anti-abbreviation instructions.
2. **v2.1 is best on field accuracy**. For applications needing accurate sub-scores (e.g. detailed ergonomic reports), v2.1 is the recommended prompt.
3. **v3 is best on final-score calibration**. For applications needing accurate risk levels (triage, screening), v3's reduced bias is valuable despite lower field accuracy.
4. **Method A vs Method B**: detailed prompts (v2.1) benefit Method A more; multi-agent (v3) benefits Method B more. A possible explanation is that v2.1's long prompt competes for attention with video frames, while v3's shorter per-agent prompt doesn't.

---

## 11. Limitations and Future Work

### Hard Limits of Gemma 4 31B

1. **Fine-grained angular discrimination** (e.g. wrist 12° vs 18°) appears to exceed the vision encoder's resolution. No amount of prompt engineering can help.
2. **Multi-person disambiguation**: 2/20 videos (REBA_10 nurse+patient, REBA_13 worker+helper) cause the model to confuse subjects. Would need subject-identification pre-processing or prompt instructions.
3. **Confidence is not a useful uncertainty signal in absolute terms** (no "low" confidence ever emitted).

### Dataset Limitations

1. **Small sample size** (44 annotations) limits statistical claims. The paper uses MAE and correlation rather than significance tests.
2. **Single annotator** — no inter-rater agreement baseline. Some fields (especially wrist) are known to have high subjectivity.
3. **Web-sourced videos** — no controlled recording conditions, varying video quality.

### Next Steps

1. **Gemini 3.1 Pro cross-model study** (in progress): Apply v1 and v2.1 prompts to the Gemini 3.1 Pro API. Expected result: absolute accuracy increases across the board (perhaps 65.9% → 75-80%), but the v1→v2.1 delta may shrink, showing that prompt engineering gains are largest on weaker models.

2. **Ground-truth review for disagreement cases**: where the model (especially v2.1 and v3) systematically disagrees with GT and the observation field contains specific angle estimates, review those annotations. Some may be true GT errors.

3. **Skeleton-overlay ablation**: augment Method A with a MediaPipe or YOLO pose-estimated skeleton overlay, test whether structured visual hints improve fine-grained angle judgments.

4. **Per-field architecture**: v3 revealed that different body parts benefit from different input methods. A hybrid architecture routing each part to its optimal input (single frame for lower_arms, video for coupling) could outperform uniform A or B.

5. **Activity score prediction** from extended video input (not a keyframe), when annotations permit.

---

## 12. Complete Code Organization

Repository layout:

```
reba-project/
├── README.md
├── requirements.txt
├── scripts/
│   └── start_vllm_server.sh
├── models/
│   └── gemma-4-31B-it/           # 59GB model weights (gitignored)
├── data/
│   ├── videos/                    # 20 MP4 files (gitignored)
│   ├── annotations/              # 44 JSON ground truth
│   └── frames/                   # ffmpeg keyframe cache (gitignored)
├── src/
│   ├── reba_tables.py            # REBA computation, Table A/B/C
│   ├── prompts/
│   │   ├── v1_baseline.py
│   │   ├── v2_detailed.py
│   │   ├── v2_1_detailed.py
│   │   └── v3_multiagent/
│   │       ├── scene_primer.py
│   │       └── part_agents.py
│   ├── experiments/
│   │   ├── client.py                 # vLLM client (sync + async)
│   │   ├── frame_extractor.py        # ffmpeg wrapper
│   │   ├── run_experiment.py         # batch runner
│   │   ├── test_single.py            # single-sample smoke test
│   │   ├── v3_orchestrator.py        # multi-agent coordinator
│   │   ├── analyze_results.py
│   │   ├── compare_versions.py
│   │   ├── inspect_observations.py
│   │   └── analyze_confidence.py
│   └── utils/
│       ├── validate_annotations.py
│       ├── validate_full_pipeline.py
│       └── fix_annotation_subscore_snapshots.py
├── results/                      # JSONL output per condition
│   └── {version}_method_{A|B}.jsonl
└── logs/                         # Analysis text outputs
```

### Reproducing Results

```bash
# 1. Setup
conda env create -f environment.yml
conda activate reba
# Download model: hf download google/gemma-4-31B-it

# 2. Start vLLM server (keep running)
./scripts/start_vllm_server.sh

# 3. In another terminal, validate data
python src/utils/validate_annotations.py
python src/utils/validate_full_pipeline.py

# 4. Run experiments (each ~30-40 minutes)
for version in v1 v2 v2_1 v3; do
  python src/experiments/run_experiment.py --prompt $version --method both
done

# 5. Generate analysis
for version in v1 v2 v2_1 v3; do
  python src/experiments/analyze_results.py --prompt $version > logs/${version}_analysis.txt
done
python src/experiments/compare_versions.py --from v1 --to v2_1 > logs/v1_vs_v2_1.txt
python src/experiments/compare_versions.py --from v2_1 --to v3 > logs/v2_1_vs_v3.txt
python src/experiments/analyze_confidence.py > logs/v3_confidence.txt
```

All experiments are deterministic (`temperature=0, seed=42`) — re-runs yield byte-identical outputs.

---

**End of walkthrough.**
