# Experimental Results

The JSONL output files (8 files, one per `prompt_version × method` pair)
are **not committed to this repository** because they embed a full snapshot
of the ground-truth annotation at inference time. As annotations may still
be revised during the project, committing results here would risk version
mismatch between paper numbers and latest labels.

## How to Regenerate

Once you have the data in place (see `../data/README.md`) and the vLLM
server running, run the full experiment suite:

```bash
python src/experiments/run_experiment.py --prompt v1   --method both
python src/experiments/run_experiment.py --prompt v2   --method both
python src/experiments/run_experiment.py --prompt v2_1 --method both
python src/experiments/run_experiment.py --prompt v3   --method both
```

This will populate this directory with:

```
results/
├── v1_method_A.jsonl
├── v1_method_B.jsonl
├── v2_method_A.jsonl
├── v2_method_B.jsonl
├── v2_1_method_A.jsonl
├── v2_1_method_B.jsonl
├── v3_method_A.jsonl
└── v3_method_B.jsonl
```

Each file contains 44 JSON lines, one per annotation. Total runtime is
~2 hours on 2×A100-40GB.

## Reproducibility

Experiments use `temperature=0.0, seed=42`. Re-running the same code on
the same data produces byte-identical outputs, provided model and vLLM
versions match.

## Record Format

See the main `../README.md` section "Input / Output Formats" for the
complete JSONL schema.

## Results Used in Paper

Paper numbers (as of 2026-04-19) were generated with Gemma 4 31B via vLLM
0.19.0 on 2×A100-40GB. If you reproduce these results on comparable
hardware with the pinned `requirements.txt`, you should obtain identical
numbers.
