"""
client.py
---------
Unified vLLM client for the REBA experiments.

Handles:
  - Method A (N keyframe images) and Method B (1 video) via OpenAI-compatible
    chat completions API (vLLM serves this at localhost:8000/v1).
  - base64 data URLs for all media (per handoff: HTTP fetch can fail on VCL).
  - Gemma 4 official recommendation: media content blocks before text.
  - Greedy + seed=42 for reproducibility (per handoff Sec 6).
  - Strict JSON parsing with enum validation against reba_tables.ENUMS;
    all parse failures are captured as structured status codes rather than
    silently coerced. One-shot: no retries (decision D3).
"""

import base64
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reba_tables import ENUMS  # noqa: E402


# =============================================================================
# Config
# =============================================================================

DEFAULT_SERVER_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL_NAME = "gemma-4-31B-it"
DEFAULT_TIMEOUT_SEC = 300   # generous; model may take a while on cold cache


# =============================================================================
# Response container
# =============================================================================

@dataclass
class InferenceResult:
    """Everything we log for a single inference call."""
    # Status: "ok" if raw output was produced and parsed to valid JSON w/ enums;
    # "parse_failed" if JSON couldn't be extracted;
    # "schema_invalid" if JSON extracted but missing fields or bad enum values;
    # "http_error" if the server call failed.
    status: str
    latency_ms: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    raw_response: str = ""
    parsed_prediction: Optional[Dict[str, Any]] = None   # the annotation-shape dict
    parse_error: Optional[str] = None                    # one-line diagnostic
    http_error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# =============================================================================
# Media encoding
# =============================================================================

def _b64_data_url(path: Path, mime: str) -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def _image_blocks(image_paths: List[Path]) -> List[dict]:
    return [
        {"type": "image_url",
         "image_url": {"url": _b64_data_url(Path(p), "image/jpeg")}}
        for p in image_paths
    ]


def _video_block(video_path: Path) -> dict:
    return {
        "type": "video_url",
        "video_url": {"url": _b64_data_url(Path(video_path), "video/mp4")},
    }


# =============================================================================
# JSON extraction from free-form model output
# =============================================================================

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?|\n?```\s*$",
                            re.IGNORECASE | re.MULTILINE)


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Pull out the first balanced {...} block from text. Ignores braces inside
    string literals. Returns the JSON substring or None.
    """
    # strip code fences first if present
    stripped = _CODE_FENCE_RE.sub("", text).strip()

    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(stripped):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return stripped[start: i + 1]
    return None


# =============================================================================
# Schema validation against reba_tables.ENUMS
# =============================================================================

_REQUIRED_SHAPE = {
    "group_a": {
        "trunk":      {"position_class", "is_twisted_sidebent"},
        "neck":       {"position_class", "is_twisted_sidebent"},
        "legs":       {"position_class", "knee_30_60_flexion", "knee_gt_60_flexion"},
    },
    "group_b": {
        "upper_arms": {"position_class", "is_abducted_rotated",
                       "is_shoulder_raised", "is_gravity_assisted"},
        "lower_arms": {"position_class"},
        "wrists":     {"position_class", "is_deviated_twisted"},
    },
    "context": {
        "load_force": {"category_class", "has_shock"},
        "coupling":   {"category_class"},
    },
}

_ENUM_KEYS = {
    ("group_a", "trunk",      "position_class"): "trunk.position_class",
    ("group_a", "neck",       "position_class"): "neck.position_class",
    ("group_a", "legs",       "position_class"): "legs.position_class",
    ("group_b", "upper_arms", "position_class"): "upper_arms.position_class",
    ("group_b", "lower_arms", "position_class"): "lower_arms.position_class",
    ("group_b", "wrists",     "position_class"): "wrists.position_class",
    ("context", "load_force", "category_class"): "load_force.category_class",
    ("context", "coupling",   "category_class"): "coupling.category_class",
}

_BOOL_FIELDS = {
    ("group_a", "trunk",      "is_twisted_sidebent"),
    ("group_a", "neck",       "is_twisted_sidebent"),
    ("group_a", "legs",       "knee_30_60_flexion"),
    ("group_a", "legs",       "knee_gt_60_flexion"),
    ("group_b", "upper_arms", "is_abducted_rotated"),
    ("group_b", "upper_arms", "is_shoulder_raised"),
    ("group_b", "upper_arms", "is_gravity_assisted"),
    ("group_b", "wrists",     "is_deviated_twisted"),
    ("context", "load_force", "has_shock"),
}


def validate_prediction(pred: dict) -> List[str]:
    """Return list of validation error messages; empty list means OK."""
    errs = []

    # structure + required fields
    for group, parts in _REQUIRED_SHAPE.items():
        if group not in pred or not isinstance(pred[group], dict):
            errs.append(f"missing/invalid group: {group}")
            continue
        for part, required_fields in parts.items():
            if part not in pred[group] or not isinstance(pred[group][part], dict):
                errs.append(f"missing/invalid part: {group}.{part}")
                continue
            present = set(pred[group][part].keys())
            missing = required_fields - present
            if missing:
                errs.append(f"{group}.{part} missing fields: {sorted(missing)}")

    # enum values
    for (group, part, field_name), enum_key in _ENUM_KEYS.items():
        try:
            v = pred[group][part][field_name]
        except (KeyError, TypeError):
            continue  # already reported above
        allowed = ENUMS[enum_key]
        if v not in allowed:
            errs.append(f"{group}.{part}.{field_name}={v!r} not in {allowed}")

    # boolean fields
    for (group, part, field_name) in _BOOL_FIELDS:
        try:
            v = pred[group][part][field_name]
        except (KeyError, TypeError):
            continue
        if not isinstance(v, bool):
            errs.append(f"{group}.{part}.{field_name}={v!r} is not a bool")

    return errs


# =============================================================================
# Top-level inference
# =============================================================================

def call_model(
    prompt_text: str,
    images: Optional[List[Path]] = None,
    video: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    server_url: str = DEFAULT_SERVER_URL,
    temperature: float = 0.0,
    seed: int = 42,
    max_tokens: int = 1024,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> InferenceResult:
    """
    Send one chat-completion request and return a structured result.

    Exactly one of `images` OR `video` must be provided.
    """
    if bool(images) == bool(video):
        raise ValueError("Provide exactly one of `images` or `video`.")

    # Build content blocks: media first (Gemma 4 convention), then text.
    content = []
    if images:
        content.extend(_image_blocks(images))
    else:
        content.append(_video_block(video))
    content.append({"type": "text", "text": prompt_text})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(server_url, json=payload, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        return InferenceResult(
            status="http_error",
            latency_ms=int((time.perf_counter() - t0) * 1000),
            http_error=f"{type(e).__name__}: {e}",
        )

    dt_ms = int((time.perf_counter() - t0) * 1000)
    data = r.json()
    raw = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    # Parse
    json_str = _extract_first_json_object(raw)
    if json_str is None:
        return InferenceResult(
            status="parse_failed",
            latency_ms=dt_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_response=raw,
            parse_error="no balanced {...} block found",
        )
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return InferenceResult(
            status="parse_failed",
            latency_ms=dt_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_response=raw,
            parse_error=f"JSONDecodeError: {e}",
        )

    errs = validate_prediction(parsed)
    if errs:
        return InferenceResult(
            status="schema_invalid",
            latency_ms=dt_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_response=raw,
            parsed_prediction=parsed,
            parse_error=" | ".join(errs),
        )

    return InferenceResult(
        status="ok",
        latency_ms=dt_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        raw_response=raw,
        parsed_prediction=parsed,
    )


# =============================================================================
# Self-test (no network call; parsing round-trip only)
# =============================================================================

if __name__ == "__main__":
    # Round-trip the parser against a valid-looking synthetic model output.
    sample_valid = """Here is my assessment:

    ```json
    {
      "group_a": {
        "trunk":  {"position_class": "upright", "is_twisted_sidebent": false},
        "neck":   {"position_class": "flex_0_to_20", "is_twisted_sidebent": false},
        "legs":   {"position_class": "bilateral_weight_bearing_or_sitting",
                   "knee_30_60_flexion": false, "knee_gt_60_flexion": false}
      },
      "group_b": {
        "upper_arms": {"position_class": "flex_45_to_90",
                       "is_abducted_rotated": false,
                       "is_shoulder_raised": false,
                       "is_gravity_assisted": false},
        "lower_arms": {"position_class": "flex_60_to_100"},
        "wrists":     {"position_class": "flex_ext_0_to_15",
                       "is_deviated_twisted": false}
      },
      "context": {
        "load_force": {"category_class": "lt_5_kg", "has_shock": false},
        "coupling":   {"category_class": "good_power_grip"}
      }
    }
    ```
    """
    js = _extract_first_json_object(sample_valid)
    assert js is not None, "failed to extract JSON from fenced output"
    parsed = json.loads(js)
    errs = validate_prediction(parsed)
    assert not errs, f"validation should pass, got: {errs}"
    print("✓ valid sample: extracted, parsed, and schema-valid")

    # Negative test: bad enum value
    bad = json.loads(js)
    bad["group_a"]["trunk"]["position_class"] = "slightly_bent"
    errs2 = validate_prediction(bad)
    assert errs2 and "trunk" in errs2[0], f"should flag bad enum, got: {errs2}"
    print(f"✓ invalid enum caught: {errs2[0]}")

    # Negative test: missing field
    bad2 = json.loads(js)
    del bad2["group_b"]["wrists"]["is_deviated_twisted"]
    errs3 = validate_prediction(bad2)
    assert any("wrists" in e and "missing" in e for e in errs3)
    print(f"✓ missing field caught: {[e for e in errs3 if 'wrists' in e][0]}")

    # Negative test: no JSON at all
    assert _extract_first_json_object("lorem ipsum no braces here") is None
    print("✓ no-JSON input correctly returns None")

    print("\nAll client.py self-tests passed.")


# =============================================================================
# Async version — for v3 multi-agent concurrent calls
# =============================================================================
# We use httpx (async HTTP client) to issue multiple simultaneous requests
# to the vLLM server. vLLM will batch them at the engine level and process
# them concurrently, so total wall time ≈ max(individual latencies) rather
# than sum. This is the key optimization that makes v3's 9-call pipeline
# tractable on a single GPU.

async def call_model_async(
    prompt_text: str,
    images=None,
    video=None,
    model_name: str = DEFAULT_MODEL_NAME,
    server_url: str = DEFAULT_SERVER_URL,
    temperature: float = 0.0,
    seed: int = 42,
    max_tokens: int = 1024,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    client=None,  # optional pre-made httpx.AsyncClient (for reuse)
) -> InferenceResult:
    """
    Async variant of call_model. Accepts an optional shared client for
    connection pooling. If `client` is None, creates a single-use one.
    """
    import httpx  # local import to avoid hard dep for synchronous users

    if bool(images) == bool(video):
        raise ValueError("Provide exactly one of `images` or `video`.")

    content = []
    if images:
        content.extend(_image_blocks(images))
    else:
        content.append(_video_block(video))
    content.append({"type": "text", "text": prompt_text})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        try:
            r = await client.post(server_url, json=payload, timeout=timeout)
            r.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            return InferenceResult(
                status="http_error",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                http_error=f"{type(e).__name__}: {e}",
            )

        dt_ms = int((time.perf_counter() - t0) * 1000)
        data = r.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
    finally:
        if owns_client:
            await client.aclose()

    # Parsing identical to sync version
    json_str = _extract_first_json_object(raw)
    if json_str is None:
        return InferenceResult(
            status="parse_failed",
            latency_ms=dt_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_response=raw,
            parse_error="no balanced {...} block found",
        )
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return InferenceResult(
            status="parse_failed",
            latency_ms=dt_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_response=raw,
            parse_error=f"JSONDecodeError: {e}",
        )

    # NOTE: we do NOT run validate_prediction() here — for v3, each call
    # returns a PARTIAL prediction (single part), and the full-prediction
    # validator expects the merged shape. The orchestrator handles merge
    # + validation.
    return InferenceResult(
        status="ok",
        latency_ms=dt_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        raw_response=raw,
        parsed_prediction=parsed,
    )


async def call_model_async_text_only(
    prompt_text: str,
    images=None,
    video=None,
    client=None,
    **kwargs,
) -> "tuple[str, int, dict]":
    """
    Thin wrapper for scene primer: returns (text, latency_ms, usage_dict)
    without JSON parsing, since the scene primer emits plain text.
    """
    import httpx
    if bool(images) == bool(video):
        raise ValueError("Provide exactly one of `images` or `video`.")

    server_url = kwargs.get("server_url", DEFAULT_SERVER_URL)
    model_name = kwargs.get("model_name", DEFAULT_MODEL_NAME)
    temperature = kwargs.get("temperature", 0.0)
    seed = kwargs.get("seed", 42)
    max_tokens = kwargs.get("max_tokens", 400)
    timeout = kwargs.get("timeout", DEFAULT_TIMEOUT_SEC)

    content = []
    if images:
        content.extend(_image_blocks(images))
    else:
        content.append(_video_block(video))
    content.append({"type": "text", "text": prompt_text})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        r = await client.post(server_url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    finally:
        if owns_client:
            await client.aclose()

    dt_ms = int((time.perf_counter() - t0) * 1000)
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, dt_ms, usage