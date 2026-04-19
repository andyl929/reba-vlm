"""
frame_extractor.py
------------------
Extract a single JPEG frame from a video at a precise timestamp using ffmpeg.

Used by Method A (keyframe-image input to the vLLM model).

Design:
  - We use ffmpeg's accurate seek (-ss after -i) to guarantee the frame at
    the specified timestamp, trading a few hundred ms of extraction time
    for timestamp precision.
  - Frames are cached under data/frames/{video_stem}_{timestamp_s}.jpg so
    we don't re-run ffmpeg every experiment.
  - Output is JPEG quality 2 (high, ~80% default quality in libjpeg).
    Higher quality makes base64 payloads bigger; 2 is a good balance.
"""

from pathlib import Path
import subprocess
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"


def _format_timestamp_for_filename(ts: float) -> str:
    """2.00 -> '2.00s', 3.47 -> '3.47s'. Stable with annotation filenames."""
    return f"{ts:.2f}s"


def extract_frame(video_path: Path, timestamp_sec: float,
                  out_dir: Path = DEFAULT_FRAMES_DIR,
                  overwrite: bool = False) -> Path:
    """
    Extract frame at timestamp_sec from video. Returns path to the JPEG.

    Parameters
    ----------
    video_path : Path to .mp4 (or similar)
    timestamp_sec : seconds (float, 2 decimal precision matches annotations)
    out_dir : where to write the JPEG (cached)
    overwrite : if False and cached file exists, return it without re-extracting
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    stem = video_path.stem
    out_name = f"{stem}_{_format_timestamp_for_filename(timestamp_sec)}.jpg"
    out_path = out_dir / out_name

    if out_path.exists() and not overwrite:
        return out_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. "
                           "Activate the 'reba' conda env first.")

    # Accurate seek: -ss AFTER -i decodes from the start but is frame-accurate.
    # For our 20s videos this adds <1s of latency; we accept it for precision.
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-ss", f"{timestamp_sec:.3f}",
        "-frames:v", "1",
        "-q:v", "2",                # jpeg quality: 1 best, 31 worst; 2 is high
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed on {video_path.name} @ {timestamp_sec}s:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {e.stderr.decode('utf-8', 'replace')}"
        ) from e

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg ran but no output frame produced for "
            f"{video_path.name} @ {timestamp_sec}s. "
            f"Is the timestamp beyond the video duration?"
        )
    return out_path


def extract_frames_for_annotations(annotations_dir: Path,
                                   videos_dir: Path,
                                   out_dir: Path = DEFAULT_FRAMES_DIR,
                                   overwrite: bool = False) -> list:
    """
    Iterate all annotation JSONs and extract their keyframes. Returns a list
    of dicts: [{annotation_file, video_file, timestamp_sec, frame_path}, ...]
    """
    import json
    results = []
    annotations_dir = Path(annotations_dir)
    videos_dir = Path(videos_dir)

    for jf in sorted(annotations_dir.glob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            ann = json.load(f)
        video_file = ann["meta_data"]["video_file"]
        ts = float(ann["meta_data"]["timestamp_sec"])
        video_path = videos_dir / video_file
        frame_path = extract_frame(video_path, ts, out_dir=out_dir,
                                   overwrite=overwrite)
        results.append({
            "annotation_file": jf.name,
            "video_file": video_file,
            "timestamp_sec": ts,
            "frame_path": str(frame_path),
        })
    return results


if __name__ == "__main__":
    # Self-test: list out what WOULD be extracted (no ffmpeg call until we
    # have real videos on VCL).
    import json
    ann_dir = PROJECT_ROOT / "data" / "annotations"
    if not ann_dir.exists():
        print(f"No annotations dir at {ann_dir}; nothing to do.")
    else:
        files = sorted(ann_dir.glob("*.json"))
        print(f"Would extract {len(files)} frames:")
        for jf in files[:5]:
            with open(jf, "r", encoding="utf-8") as f:
                ann = json.load(f)
            print(f"  {ann['meta_data']['video_file']} @ "
                  f"{ann['meta_data']['timestamp_sec']}s -> "
                  f"{jf.stem}.jpg")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")