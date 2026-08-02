"""End-to-end latency measurements from a session CSV.

Every frame processed by main.py logs three timings in the same row:
  - engagement_latency_ms : webcam frame -> face landmarker -> yaw/pitch
  - yolo_latency_ms        : object-detection inference (only on scan frames;
                              blank on the ~29 out of 30 frames it doesn't run)
  - loop_latency_ms        : the full per-frame budget (capture through
                              compositing), i.e. what actually gates framerate

Usage: python -m eval.latency_eval [path/to/session_*.csv]
       (defaults to the most recent file in logs/)
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import config


def _latest_session_csv() -> Path | None:
    candidates = sorted(glob.glob(str(config.LOGS_DIR / "session_*.csv")))
    return Path(candidates[-1]) if candidates else None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else float("nan"),
    }


def main(path_arg: str | None = None) -> None:
    path = Path(path_arg) if path_arg else _latest_session_csv()
    if path is None or not path.exists():
        print("No session CSV found. Run: python main.py --label  (or --test-frames N)")
        return

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{path} is empty.")
        return

    stages = {
        "engagement (frame -> yaw/pitch)": "engagement_latency_ms",
        "memory scan (YOLO, when it runs)": "yolo_latency_ms",
        "full per-frame loop": "loop_latency_ms",
    }

    print(f"Latency -- {path.name} ({len(rows)} frames)")
    for label, col in stages.items():
        values = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        s = summarize(values)
        if s["n"] == 0:
            print(f"  {label}: no samples")
            continue
        print(f"  {label}: n={s['n']}  p50={s['p50']:.1f}ms  p95={s['p95']:.1f}ms  "
              f"p99={s['p99']:.1f}ms  max={s['max']:.1f}ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
