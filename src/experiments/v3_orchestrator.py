"""
v3_orchestrator.py
------------------
Coordinates the v3 multi-agent pipeline:
  1. One scene primer call (plain-text output)
  2. Eight parallel part-agent calls (JSON output each)
  3. Merge partials into full annotation shape
  4. Validate against ENUMS
  5. Compute REBA scores via reba_tables

Concurrency strategy: we use concurrent.futures.ThreadPoolExecutor to fire
off the 8 part agents in parallel via the synchronous `requests` library.
vLLM accepts multiple simultaneous HTTP connections and batches them at
the engine level, so total wall time ≈ slowest single call, not sum.
We avoid httpx to keep the dependency footprint identical to v1/v2/v2.1.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.client import (                                      # noqa: E402
    call_model, validate_prediction,
    DEFAULT_SERVER_URL, DEFAULT_MODEL_NAME, DEFAULT_TIMEOUT_SEC,
)
from prompts.v3_multiagent.scene_primer import build_scene_primer_prompt  # noqa: E402
from prompts.v3_multiagent.part_agents import (                       # noqa: E402
    build_full_part_prompt, PART_ORDER, PART_TO_GROUP, PART_CONFIGS,
)


@dataclass
class V3Result:
    """Result of one complete v3 multi-agent evaluation."""
    # Same four statuses as single-call InferenceResult
    status: str   # "ok" | "schema_invalid" | "parse_failed" | "http_error"
    latency_ms: int                          # wall-clock time for the whole pipeline
    parsed_prediction: Optional[dict] = None  # merged prediction in annotation shape
    parse_error: Optional[str] = None
    http_error: Optional[str] = None

    # v3-specific metadata
    scene_description: str = ""
    scene_latency_ms: int = 0
    scene_tokens: Dict[str, int] = field(default_factory=dict)
    per_part_latencies_ms: Dict[str, int] = field(default_factory=dict)
    per_part_tokens: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_part_status: Dict[str, str] = field(default_factory=dict)
    per_part_confidence: Dict[str, str] = field(default_factory=dict)
    per_part_raw: Dict[str, str] = field(default_factory=dict)


def _merge_partials(partials: Dict[str, dict]) -> dict:
    """
    Given {part_name: {enum_field, confidence, adjustments...}},
    produce a full annotation-shaped dict:
      {"group_a": {...}, "group_b": {...}, "context": {...}}
    Drops the 'confidence' field from each part (it lives in V3Result
    metadata instead, not in the prediction itself).
    """
    out = {"group_a": {}, "group_b": {}, "context": {}}
    for part_name in PART_ORDER:
        partial = partials.get(part_name)
        if partial is None:
            continue
        group = PART_TO_GROUP[part_name]
        # Strip confidence, keep everything else
        cleaned = {k: v for k, v in partial.items() if k != "confidence"}
        out[group][part_name] = cleaned
    return out


def _call_scene_primer(image_paths, video_path, method, timestamp,
                       server_url, model_name):
    """Call scene primer and return (text, latency_ms, usage_dict).
    Uses the existing sync call_model but ignores JSON parsing (scene
    primer returns plain text)."""
    import requests
    import base64

    primer_prompt = build_scene_primer_prompt(method, timestamp)

    content = []
    if image_paths:
        for p in image_paths:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    else:
        with open(video_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{b64}"},
        })
    content.append({"type": "text", "text": primer_prompt})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 400,
    }
    t0 = time.perf_counter()
    r = requests.post(server_url, json=payload, timeout=DEFAULT_TIMEOUT_SEC)
    r.raise_for_status()
    dt_ms = int((time.perf_counter() - t0) * 1000)
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, dt_ms, usage


def _call_one_part(part_name, scene_text, image_paths, video_path,
                   method, timestamp, server_url, model_name):
    """Worker function for a single part agent. Runs in a thread."""
    prompt = build_full_part_prompt(part_name, scene_text, method, timestamp)
    result = call_model(
        prompt,
        images=image_paths,
        video=video_path,
        server_url=server_url,
        model_name=model_name,
        max_tokens=300,
    )
    # NOTE: call_model runs validate_prediction() which assumes full schema.
    # For a single-part partial prediction, that validator will always fail.
    # We demote any "schema_invalid" to "ok" here as long as the JSON parsed;
    # full-prediction validation happens after merge in the orchestrator.
    if result.status == "schema_invalid" and result.parsed_prediction:
        result.status = "ok"
        result.parse_error = None
    return part_name, result


def _run_v3_sync(image_paths=None, video_path=None,
                 method: str = "A", timestamp: float = None,
                 server_url: str = DEFAULT_SERVER_URL,
                 model_name: str = DEFAULT_MODEL_NAME) -> "V3Result":
    """Synchronous orchestrator. Scene primer + 8 parallel part agents."""
    overall_t0 = time.perf_counter()

    # Step 1: Scene primer (serial)
    try:
        scene_text, scene_dt, scene_usage = _call_scene_primer(
            image_paths, video_path, method, timestamp,
            server_url, model_name,
        )
    except Exception as e:
        return V3Result(
            status="http_error",
            latency_ms=int((time.perf_counter() - overall_t0) * 1000),
            http_error=f"scene_primer: {type(e).__name__}: {e}",
        )

    # Step 2: 8 part agents in parallel via thread pool
    part_results = {}
    with ThreadPoolExecutor(max_workers=len(PART_ORDER)) as pool:
        futures = [
            pool.submit(
                _call_one_part,
                part, scene_text, image_paths, video_path,
                method, timestamp, server_url, model_name,
            )
            for part in PART_ORDER
        ]
        for fut in futures:
            part_name, r = fut.result()
            part_results[part_name] = r

    total_dt_ms = int((time.perf_counter() - overall_t0) * 1000)

    result = V3Result(
        status="ok",
        latency_ms=total_dt_ms,
        scene_description=scene_text,
        scene_latency_ms=scene_dt,
        scene_tokens={
            "prompt": scene_usage.get("prompt_tokens"),
            "completion": scene_usage.get("completion_tokens"),
        },
    )

    partials = {}
    any_http_error = False
    any_parse_failed = False

    for part_name, r in part_results.items():
        result.per_part_latencies_ms[part_name] = r.latency_ms
        result.per_part_tokens[part_name] = {
            "prompt": r.prompt_tokens,
            "completion": r.completion_tokens,
        }
        result.per_part_status[part_name] = r.status
        result.per_part_raw[part_name] = r.raw_response

        if r.status == "http_error":
            any_http_error = True
            continue
        if r.status == "parse_failed":
            any_parse_failed = True
            continue
        partials[part_name] = r.parsed_prediction or {}
        conf = (r.parsed_prediction or {}).get("confidence")
        if conf:
            result.per_part_confidence[part_name] = conf

    merged = _merge_partials(partials)
    result.parsed_prediction = merged

    if any_http_error:
        result.status = "http_error"
        result.http_error = "one or more part calls failed HTTP"
    elif any_parse_failed:
        result.status = "parse_failed"
        result.parse_error = (
            "parts failed JSON parse: "
            + ", ".join(p for p, s in result.per_part_status.items()
                        if s == "parse_failed")
        )
    else:
        errs = validate_prediction(merged)
        if errs:
            result.status = "schema_invalid"
            result.parse_error = " | ".join(errs)

    return result


def run_v3_single(image_paths=None, video_path=None,
                  method: str = "A", timestamp: float = None,
                  server_url: str = DEFAULT_SERVER_URL,
                  model_name: str = DEFAULT_MODEL_NAME) -> "V3Result":
    """Synchronous entry point used by run_experiment.py."""
    return _run_v3_sync(
        image_paths=image_paths, video_path=video_path,
        method=method, timestamp=timestamp,
        server_url=server_url, model_name=model_name,
    )


def to_dict(result: V3Result) -> dict:
    """Convert V3Result to dict for JSONL logging (matches normal shape
    plus v3-specific metadata)."""
    d = {
        "status": result.status,
        "latency_ms": result.latency_ms,
        "prompt_tokens": None,      # not meaningful for multi-agent; see per-part
        "completion_tokens": None,
        "raw_response": "",         # per-part raw responses are captured separately
        "parsed_prediction": result.parsed_prediction,
        "parse_error": result.parse_error,
        "http_error": result.http_error,
        "v3_meta": {
            "scene_description": result.scene_description,
            "scene_latency_ms": result.scene_latency_ms,
            "scene_tokens": result.scene_tokens,
            "per_part_latencies_ms": result.per_part_latencies_ms,
            "per_part_tokens": result.per_part_tokens,
            "per_part_status": result.per_part_status,
            "per_part_confidence": result.per_part_confidence,
            "per_part_raw": result.per_part_raw,
        },
    }
    return d


if __name__ == "__main__":
    # Quick import smoke test (no network call)
    print("PART_ORDER:", PART_ORDER)
    print("Number of part configs:", len(PART_CONFIGS))
    print("\n✓ orchestrator imports OK. Call run_v3_single() with real inputs.")