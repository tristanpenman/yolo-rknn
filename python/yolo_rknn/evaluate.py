"""Run repeatable YOLOv5 evaluation and export metrics/plots.

This script wraps ``yolov5/val.py`` to make evaluation reproducible and to
collect key outputs (metrics + plot images) in a single directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


PLOT_FILES = [
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "confusion_matrix.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv5 eval and export artifacts")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--data", required=True, help="Path to dataset yaml")
    parser.add_argument("--img", type=int, default=640, help="Inference image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--device", default="", help="CUDA device or cpu")
    parser.add_argument("--project", default="yolov5/runs/val", help="YOLOv5 project output dir")
    parser.add_argument("--name", default="repeatable-eval", help="YOLOv5 run name")
    parser.add_argument(
        "--export-dir",
        default="reports/evaluation",
        help="Directory where metrics and plots are copied",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Reuse existing YOLOv5 run name instead of auto-incrementing",
    )
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> Path:
    cmd: List[str] = [
        "python3",
        "thirdparty/yolov5/val.py",
        "--weights",
        args.weights,
        "--data",
        args.data,
        "--img",
        str(args.img),
        "--batch",
        str(args.batch),
        "--project",
        args.project,
        "--name",
        args.name,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.exist_ok:
        cmd.append("--exist-ok")

    subprocess.run(cmd, check=True)
    return Path(args.project) / args.name


def load_metrics(results_csv: Path) -> Dict[str, float]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {results_csv}")

    latest = rows[-1]
    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "fitness",
    ]
    metrics: Dict[str, float] = {}
    for key in keys:
        if key in latest and latest[key] != "":
            metrics[key] = float(latest[key])
    return metrics


def copy_artifacts(run_dir: Path, export_dir: Path) -> List[str]:
    copied: List[str] = []
    export_dir.mkdir(parents=True, exist_ok=True)

    for filename in PLOT_FILES:
        src = run_dir / filename
        if src.exists():
            dst = export_dir / filename
            shutil.copy2(src, dst)
            copied.append(filename)

    labels_png = run_dir / "labels.jpg"
    if labels_png.exists():
        dst = export_dir / labels_png.name
        shutil.copy2(labels_png, dst)
        copied.append(labels_png.name)

    return copied


def main() -> None:
    args = parse_args()
    run_dir = run_eval(args)

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results.csv in {run_dir}")

    metrics = load_metrics(results_csv)
    export_dir = Path(args.export_dir)
    copied_files = copy_artifacts(run_dir, export_dir)

    metrics_path = export_dir / "metrics.json"
    report = {
        "weights": args.weights,
        "data": args.data,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "copied_plots": copied_files,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    results_copy = export_dir / "results.csv"
    shutil.copy2(results_csv, results_copy)

    print(f"Evaluation artifacts exported to: {export_dir}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
