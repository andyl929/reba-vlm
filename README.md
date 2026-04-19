# REBA-VLM: Ergonomic Risk Assessment with Multimodal LLMs

Course project — NCSU ISE Spring 2026

This repository contains code, prompts, and experimental results for the project **"Prompt Engineering and Multi-Agent Decomposition for Ergonomic Risk Assessment from Video on Gemma 4 31B."** We investigate whether a local multimodal large language model can perform Rapid Entire Body Assessment (REBA) — a standard ergonomic risk scoring tool — directly from worker video or keyframe input, through iterative prompt design and multi-agent decomposition.

---

## Headline Results

| Metric | v1 baseline | v2 detailed | v2.1 refined | v3 multi-agent |
|---|---|---|---|---|
| Schema compliance (Method A) | 36.4% | **100%** | 100% | 100% |
| Field-level accuracy (Method A) | 60.3% | 62.7% | **65.9%** | 63.5% |
| Final REBA MAE (Method B) | 2.40 | 2.64 | 2.77 | **2.43** |
| Final REBA bias (Method B) | −0.50 | −1.41 | −0.77 | **−0.30** |
| Final REBA Pearson r (Method B) | 0.25 | 0.21 | 0.03 | **0.27** |

Best per column in **bold**. See `project_walkthrough.md` for complete analysis.

---

## Code Organization

```
reba-vlm/
├── README.md                           # This file
├── LICENSE                             # MIT
├── requirements.txt                    # Python dependencies
├── project_walkthrough.md              # Full project narrative with all findings
│
├── scripts/
│   └── start_vllm_server.sh            # Launch vLLM with Gemma 4 31B
│
├── src/
│   ├── reba_tables.py                  # REBA Tables A/B/C, sub-score computation
│   │
│   ├── prompts/                        # Four prompt architectures
│   │   ├── v1_baseline.py              # Minimal REBA-worksheet prompt
│   │   ├── v2_detailed.py              # + observation scaffolding + angular rules
│   │   ├── v2_1_detailed.py            # v2 with wrist-fix + per-enum reminders
│   │   └── v3_multiagent/
│   │       ├── __init__.py
│   │       ├── scene_primer.py         # Scene-primer agent prompt
│   │       └── part_agents.py          # 8 part-specific agent prompts
│   │
│   ├── experiments/                    # Experiment runners & analyzers
│   │   ├── client.py                   # vLLM HTTP client (sync + async)
│   │   ├── frame_extractor.py          # ffmpeg keyframe extraction (Method A)
│   │   ├── test_single.py              # Single-sample end-to-end smoke test
│   │   ├── run_experiment.py           # Batch runner with resume support
│   │   ├── v3_orchestrator.py          # Multi-agent coordinator
│   │   ├── analyze_results.py          # Per-version analysis (accuracy, confusion)
│   │   ├── compare_versions.py         # Side-by-side version comparison
│   │   ├── inspect_observations.py     # Observation-field quality metrics
│   │   └── analyze_confidence.py       # v3-only confidence vs accuracy
│   │
│   └── utils/
│       ├── validate_annotations.py              # Sub-score consistency check
│       ├── validate_full_pipeline.py            # End-to-end score verification
│       └── fix_annotation_subscore_snapshots.py # One-time annotation repair
│
├── data/                               # See data/README.md
│   └── README.md                       # (videos and annotations not committed)
│
└── results/                            # JSONL outputs (NOT committed; regenerate per § 6)
    └── README.md                       # How to regenerate
```

### Folder Purposes

| Folder | Purpose |
|---|---|
| `src/prompts/` | Prompt templates. Each is a standalone Python module exporting `build_prompt(method, timestamp=None)`. v3 prompts are split across `scene_primer.py` and `part_agents.py`. |
| `src/experiments/` | Experiment runners and analyzers. The runner (`run_experiment.py`) wraps everything; the analyzers read the JSONL output files. |
| `src/utils/` | One-shot data-integrity scripts. |
| `results/` | JSONL files — one line per `(annotation, prompt_version, method)` tuple. |
| `scripts/` | Shell scripts for infrastructure (launching vLLM). |

### Not in the Repository

- **`data/videos/`** — 20 MP4 workplace videos (291 MB, licensing review pending redistribution)
- **`data/annotations/`** — 44 ground-truth JSON annotations (may be revised)
- **`models/`** — Gemma 4 31B weights (59 GB, download via `hf`)
- **`logs/`** — vLLM server logs and analysis text outputs (regenerated per run)
- **`results/*.jsonl`** — raw inference outputs (regenerate via step 6 below; not committed because they embed a snapshot of ground truth at inference time)

See `data/README.md` and `results/README.md` for further details.

---

## Running Instructions

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n reba python=3.12 -y
conda activate reba

# Install Python packages
pip install -r requirements.txt

# Install ffmpeg (for keyframe extraction)
conda install -c conda-forge ffmpeg -y
```

**Hardware requirements**: 2× NVIDIA A100-40GB (or equivalent) for tensor-parallel vLLM inference. Single-GPU setups will require lowering `--max-model-len` and may OOM on Method B (video input).

### 2. Download the Model

```bash
hf auth login   # or: huggingface-cli login
hf download google/gemma-4-31B-it --local-dir ./models/gemma-4-31B-it
```

Expect ~59 GB download, ~15 minutes on a 65 MB/s link.

### 3. Set up Data

```
data/
├── videos/                        # your .mp4 files
│   ├── REBA_1.mp4
│   ├── REBA_2.mp4
│   └── ...
└── annotations/                   # your ground-truth JSONs
    ├── REBA_REBA_1_2.00s.json
    ├── REBA_REBA_10_1.70s.json
    └── ...
```

Annotation JSON schema is documented in `project_walkthrough.md` Section 2. Our 44 annotations were produced with a custom Streamlit tool (not included in this repo).

### 4. Verify Data Integrity

```bash
python src/utils/validate_annotations.py
python src/utils/validate_full_pipeline.py
```

Expected output: "ALL 352 sub-scores reproducible" and "44/44 annotations match computed pipeline". If either fails, your annotations may be an older revision.

### 5. Start vLLM Server

```bash
./scripts/start_vllm_server.sh
```

The server takes 5-10 minutes to load. Wait for `Uvicorn running on http://0.0.0.0:8000` before proceeding.

Verify:
```bash
curl http://localhost:8000/health
```

### 6. Run Experiments

```bash
# Single-sample smoke test (recommended first)
python src/experiments/test_single.py --prompt v2_1 --method A

# Limited smoke test (3 annotations × 2 methods)
python src/experiments/run_experiment.py --prompt v2_1 --method both --limit 3

# Full runs (auto-resume if interrupted)
python src/experiments/run_experiment.py --prompt v1   --method both
python src/experiments/run_experiment.py --prompt v2   --method both
python src/experiments/run_experiment.py --prompt v2_1 --method both
python src/experiments/run_experiment.py --prompt v3   --method both

# Force re-run (deletes existing results)
python src/experiments/run_experiment.py --prompt v3 --method both --force
```

Approximate full-pipeline runtime on 2×A100-40GB:
- v1: ~20 min
- v2: ~25 min
- v2.1: ~30 min
- v3: ~50 min (9 model calls per sample)
- **Total: ~2 hours** for complete replication

### 7. Analyze Results

```bash
# Per-version analysis
python src/experiments/analyze_results.py --prompt v2_1 > logs/v2_1_analysis.txt

# Side-by-side version comparison
python src/experiments/compare_versions.py --from v1   --to v2_1 > logs/v1_vs_v2_1.txt
python src/experiments/compare_versions.py --from v2_1 --to v3   > logs/v2_1_vs_v3.txt

# Observation-field quality (v2+)
python src/experiments/inspect_observations.py --prompt v2_1 > logs/v2_1_observations.txt

# v3-only confidence analysis
python src/experiments/analyze_confidence.py --method both > logs/v3_confidence_analysis.txt
```

---

## Input / Output Formats

### Experiment Input (per annotation)

- **Video file**: `data/videos/<video_name>.mp4`
- **Annotation file**: `data/annotations/<file>.json` with required fields `meta_data`, `group_a`, `group_b`, `context`, `scores`

### Experiment Output (JSONL, one line per annotation)

```json
{
  "annotation_file": "REBA_REBA_1_2.00s.json",
  "video_file": "REBA_1.mp4",
  "timestamp_sec": 2.00,
  "prompt_version": "v2_1",
  "method": "A",
  "ground_truth": { ... full annotation JSON ... },
  "inference": {
    "status": "ok" | "schema_invalid" | "parse_failed" | "http_error",
    "latency_ms": 18216,
    "prompt_tokens": 3661,
    "completion_tokens": 606,
    "raw_response": "...",
    "parsed_prediction": { ... annotation-shaped dict ... },
    "parse_error": null,
    "http_error": null
  },
  "computed_scores": {
    "table_a": 6, "score_a": 8,
    "table_b": 7, "score_b": 8,
    "score_c": 10, "activity_score": 0,
    "final_reba_score": 10,
    "action_level": {"level": 3, "risk": "High", "action_required": "Necessary soon"}
  }
}
```

v3 adds a `v3_meta` field inside `inference` with `scene_description`, `per_part_latencies_ms`, `per_part_tokens`, `per_part_confidence`, etc.

### Determinism

All experiments use `temperature=0.0, seed=42`. Re-runs are byte-identical.

---

## Configuration

Key parameters (`scripts/start_vllm_server.sh`):

```
--tensor-parallel-size 2          # number of GPUs
--max-model-len 8192              # context window (lower if OOM)
--gpu-memory-utilization 0.90     # lower to share GPU
--dtype bfloat16
```

Key parameters (`src/experiments/client.py`):

```
DEFAULT_SERVER_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL_NAME = "gemma-4-31B-it"
DEFAULT_TIMEOUT_SEC = 300
```

---

## Citation

```
Li, A. (2026). Prompt Engineering and Multi-Agent Decomposition for Ergonomic
Risk Assessment from Video: A Study on Gemma 4 31B. Course project report,
NCSU Industrial & Systems Engineering, Spring 2026.
```

Original REBA reference:
```
Hignett, S., & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA).
Applied Ergonomics, 31(2), 201–205.
```

---

## License

MIT — see `LICENSE`.

## Contact

Andy Li — ali35@ncsu.edu
NCSU Department of Industrial & Systems Engineering