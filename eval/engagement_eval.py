"""Engagement detection reliability metrics from a labeled session CSV.

The CSV comes from `python main.py --label`: while it runs, press SPACE
to toggle a ground-truth flag ("looking at the lamp" / "away") to match
what you're actually doing. Every frame logs both the detector's
prediction and your ground truth, so this is just confusion-matrix
arithmetic over a real session.

One catch: human reaction time (~200-500ms) plus the detector's own
intentional dwell latency (~150-250ms, see config.ENGAGE_*_DWELL_FRAMES)
means every gaze transition produces a run of frames where label and
prediction are correctly out of sync, not wrongly. Scoring those as
errors just punishes the hysteresis I added on purpose. So this reports
both raw numbers and a windowed version that drops frames within
EXCLUSION_WINDOW_MS of a label change.

Usage: python -m eval.engagement_eval [path/to/session_*.csv]
       (defaults to the most recent file in logs/)
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import config

EXCLUSION_WINDOW_MS = 500.0


def _latest_session_csv() -> Path | None:
    candidates = sorted(glob.glob(str(config.LOGS_DIR / "session_*.csv")))
    return Path(candidates[-1]) if candidates else None


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def label_change_timestamps(rows: list[dict]) -> list[float]:
    timestamps = []
    prev = None
    for r in rows:
        label = int(r["label"])
        if prev is not None and label != prev:
            timestamps.append(float(r["t_ms"]))
        prev = label
    return timestamps


def exclude_near_label_transitions(rows: list[dict], window_ms: float = EXCLUSION_WINDOW_MS) -> list[dict]:
    transition_ts = label_change_timestamps(rows)
    if not transition_ts:
        return rows
    kept = []
    for r in rows:
        t = float(r["t_ms"])
        if any(abs(t - tt) < window_ms for tt in transition_ts):
            continue
        kept.append(r)
    return kept


def confusion_matrix(rows: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        pred = int(r["engaged_pred"])
        label = int(r["label"])
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_confusion(cm: dict) -> dict:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "n": total}


def flicker_rate(rows: list[dict]) -> float:
    """Predicted-state transitions per minute -- a proxy for how jittery the
    detector is, independent of whether it's *correct*. A detector could
    have decent accuracy while flickering constantly right at a boundary;
    this catches that failure mode separately from precision/recall.
    Duration comes from the session's own t_ms column, not an assumed
    framerate -- webcam capture doesn't hold a fixed fps."""
    if len(rows) < 2:
        return 0.0
    transitions = sum(
        1 for a, b in zip(rows, rows[1:])
        if int(a["engaged_pred"]) != int(b["engaged_pred"])
    )
    duration_min = max((float(rows[-1]["t_ms"]) - float(rows[0]["t_ms"])) / 1000.0 / 60.0, 1e-6)
    return transitions / duration_min


def _print_report(title: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {title}: no frames in this window")
        return
    cm = confusion_matrix(rows)
    m = metrics_from_confusion(cm)
    print(f"  {title} (n={m['n']}):")
    print(f"    confusion matrix: TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")
    print(f"    precision: {m['precision']:.3f}  recall: {m['recall']:.3f}  "
          f"f1: {m['f1']:.3f}  accuracy: {m['accuracy']:.3f}")


def main(path_arg: str | None = None) -> None:
    path = Path(path_arg) if path_arg else _latest_session_csv()
    if path is None or not path.exists():
        print("No session CSV found. Run: python main.py --label")
        return

    rows = load_rows(path)
    if not rows:
        print(f"{path} is empty.")
        return

    windowed = exclude_near_label_transitions(rows)

    print(f"Engagement detection reliability -- {path.name} ({len(rows)} frames)")
    _print_report("raw (includes reaction-time/dwell mismatch near transitions)", rows)
    _print_report(f"windowed (excludes frames within {EXCLUSION_WINDOW_MS:.0f}ms of a label change)", windowed)
    print(f"  flicker: {flicker_rate(rows):.1f} predicted-state changes/min")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
