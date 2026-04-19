# Data Directory

The raw videos (`videos/`) and ground-truth annotations (`annotations/`) are
**not committed to this repository**:

- Videos are sourced from public workplace training footage, but redistributing
  them requires individual licensing review.
- Annotations may be revised as the project progresses (e.g. after cross-model
  review with Gemini Pro), so pinning a snapshot to the repo could cause
  confusion between paper results and latest labels.

## Expected Layout

```
data/
├── videos/
│   ├── REBA_1.mp4
│   ├── REBA_2.mp4
│   ├── ...
│   └── REBA_20.MP4              # note: last file has uppercase extension
└── annotations/
    ├── REBA_REBA_1_2.00s.json
    ├── REBA_REBA_10_1.70s.json
    └── ...                       # 44 JSON files total
```

## Annotation Schema

Each annotation JSON has the structure documented in `../project_walkthrough.md`,
Section 2. Short summary:

```json
{
  "meta_data": { "video_file": "...", "timestamp_sec": 2.00, ... },
  "group_a":  { "trunk": {...}, "neck": {...}, "legs": {...} },
  "group_b":  { "upper_arms": {...}, "lower_arms": {...}, "wrists": {...} },
  "context":  { "load_force": {...}, "coupling": {...} },
  "scores":   { "table_a": ..., "final_reba_score": ..., "action_level": {...} }
}
```

## Getting the Data

Contact the author (ali35@ncsu.edu) for access.

## Verifying Your Data

Once data is in place, run:

```bash
python src/utils/validate_annotations.py
python src/utils/validate_full_pipeline.py
```

Expected output: `352/352 sub-scores reproducible` and `44/44 annotations match
computed pipeline`.
