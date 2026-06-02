"""Download Open Images examples and write YOLOv5 datasets.

Adapted from the downloader tool in OIDv6_Toolkit:
https://github.com/Bukkster/OIDv6_ToolKit

Originally released under the GPL 3.0 license.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm


SPLITS = ("train", "validation", "test")
CLASS_DESCRIPTIONS = "class-descriptions-boxable.csv"
ANNOTATION_FILES = {
    "train": "train-annotations-bbox.csv",
    "validation": "validation-annotations-bbox.csv",
    "test": "test-annotations-bbox.csv",
}
METADATA_URLS = {
    CLASS_DESCRIPTIONS: (
        "https://storage.googleapis.com/openimages/2018_04/"
        "class-descriptions-boxable.csv"
    ),
    "train-annotations-bbox.csv": (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-train-annotations-bbox.csv"
    ),
    "validation-annotations-bbox.csv": (
        "https://storage.googleapis.com/openimages/v5/"
        "validation-annotations-bbox.csv"
    ),
    "test-annotations-bbox.csv": (
        "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"
    ),
}


@dataclass(frozen=True)
class DownloadResult:
    image_id: str
    split: str
    url: str
    status: str
    error: str = ""


@dataclass
class RunSummary:
    downloaded_images: int = 0
    skipped_existing_images: int = 0
    failed_downloads: int = 0
    labels_written: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Open Images data and write YOLOv5 labels."
    )
    parser.add_argument("dataset", help="Dataset name, for example apples-oranges")
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Open Images class names in YOLO class ID order",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=(*SPLITS, "all"),
        help="Dataset splits to download",
    )
    parser.add_argument("--output-root", default="datasets", type=Path)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        help="Optional directory to use as the Open Images metadata cache",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of images per split after class filtering",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Concurrent image download count",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing dataset directory and YAML file",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Download missing metadata without prompting",
    )
    args = parser.parse_args(argv)
    args.splits = normalize_splits(args.splits)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")
    if args.workers < 1:
        parser.error("--workers must be greater than zero")
    return args


def normalize_splits(splits: Iterable[str]) -> list[str]:
    selected = list(splits)
    if "all" in selected:
        return list(SPLITS)
    return list(dict.fromkeys(selected))


def required_metadata_files(splits: Iterable[str]) -> list[str]:
    files = [CLASS_DESCRIPTIONS]
    files.extend(ANNOTATION_FILES[split] for split in splits)
    return files


def confirm_metadata_download(missing_files: list[Path], assume_yes: bool) -> None:
    if not missing_files or assume_yes:
        return
    file_list = "\n".join(f"  - {path}" for path in missing_files)
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Missing Open Images metadata files:\n"
            f"{file_list}\n"
            "Re-run with --yes to download them non-interactively."
        )
    response = input(
        "Download missing Open Images metadata files?\n"
        f"{file_list}\n"
        "Continue [y/N]? "
    )
    if response.strip().lower() not in {"y", "yes"}:
        raise RuntimeError("Metadata download cancelled")


def ensure_metadata(csv_dir: Path, splits: Iterable[str], assume_yes: bool = False) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    required_files = required_metadata_files(splits)
    missing = [csv_dir / name for name in required_files if not (csv_dir / name).exists()]
    confirm_metadata_download(missing, assume_yes)
    for name in required_files:
        path = csv_dir / name
        if path.exists():
            continue
        download_file(METADATA_URLS[name], path, show_progress=True)


def download_file(
    url: str, destination: Path, timeout: int = 30, show_progress: bool = False
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination.name,
            disable=not show_progress,
            leave=False,
        )
        temp_file_cm = NamedTemporaryFile(
            "wb",
            delete=False,
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
        with temp_file_cm as (temp_file, progress):
            temp_path = Path(temp_file.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temp_file.write(chunk)
                    progress.update(len(chunk))
    temp_path.replace(destination)


def copy_metadata(csv_dir: Path, annotations_dir: Path, splits: Iterable[str]) -> None:
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for name in required_metadata_files(splits):
        source = csv_dir / name
        destination = annotations_dir / name
        if source.resolve() == destination.resolve():
            continue
        shutil.copy2(source, destination)


def load_class_map(csv_dir: Path) -> dict[str, str]:
    class_map: dict[str, str] = {}
    with (csv_dir / CLASS_DESCRIPTIONS).open(newline="", encoding="utf-8") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 2:
                class_map[row[1]] = row[0]
    return class_map


def resolve_class_codes(class_map: dict[str, str], classes: list[str]) -> dict[str, str]:
    missing = [class_name for class_name in classes if class_name not in class_map]
    if missing:
        available_hint = ", ".join(sorted(class_map)[:10])
        raise ValueError(
            "Unknown Open Images class name(s): "
            f"{', '.join(missing)}. Check spelling and capitalization. "
            f"Examples from metadata: {available_hint}"
        )
    return {class_name: class_map[class_name] for class_name in classes}


def select_annotations(
    annotations: pd.DataFrame, class_codes: Iterable[str], limit: int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    selected = annotations[annotations["LabelName"].isin(set(class_codes))].copy()
    if selected.empty:
        return selected, []

    image_ids = list(dict.fromkeys(selected["ImageID"].astype(str)))
    if limit is not None:
        image_ids = image_ids[:limit]
        selected = selected[selected["ImageID"].isin(image_ids)].copy()
    return selected, image_ids


def write_yolo_labels(
    annotations: pd.DataFrame, class_ids: dict[str, int], labels_dir: Path
) -> int:
    labels_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for image_id, rows in annotations.groupby("ImageID", sort=False):
        label_path = labels_dir / f"{image_id}.txt"
        lines = []
        for _, row in rows.iterrows():
            x_min = float(row["XMin"])
            x_max = float(row["XMax"])
            y_min = float(row["YMin"])
            y_max = float(row["YMax"])
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            lines.append(
                f"{class_ids[row['LabelName']]} "
                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
        if not lines:
            continue
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
    return written


def image_url(split: str, image_id: str) -> str:
    return f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"


def download_one_image(image_id: str, split: str, images_dir: Path) -> DownloadResult:
    url = image_url(split, image_id)
    destination = images_dir / f"{image_id}.jpg"
    if destination.exists():
        return DownloadResult(image_id, split, url, "skipped")
    try:
        download_file(url, destination)
    except requests.RequestException as exc:
        return DownloadResult(image_id, split, url, "failed", str(exc))
    except OSError as exc:
        return DownloadResult(image_id, split, url, "failed", str(exc))
    return DownloadResult(image_id, split, url, "downloaded")


def download_images(
    image_ids: Iterable[str], split: str, images_dir: Path, workers: int
) -> list[DownloadResult]:
    images_dir.mkdir(parents=True, exist_ok=True)
    image_ids = list(image_ids)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_one_image, image_id, split, images_dir)
            for image_id in image_ids
        ]
        return [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Images ({split})",
                unit="img",
            )
        ]


def remove_failed_labels(results: Iterable[DownloadResult], labels_dir: Path) -> None:
    for result in results:
        if result.status == "failed":
            (labels_dir / f"{result.image_id}.txt").unlink(missing_ok=True)


def write_dataset_yaml(yaml_path: Path, dataset_dir: Path, classes: list[str]) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yolov5_dir = Path.cwd() / "yolov5"
    dataset_path = Path(
        os.path.relpath(dataset_dir.resolve(), start=yolov5_dir.resolve())
    ).as_posix()
    lines = [
        f"path: {dataset_path}",
        "train: images/train",
        "test: images/test",
        "val: images/validation",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {class_name}" for index, class_name in enumerate(classes))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output_paths(
    output_root: Path, dataset: str, overwrite: bool
) -> tuple[Path, Path, Path]:
    dataset_dir = output_root / dataset
    yaml_path = output_root / f"{dataset}.yaml"
    annotations_dir = dataset_dir / "annotations"
    if overwrite:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        yaml_path.unlink(missing_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, yaml_path, annotations_dir


def write_error_report(dataset_dir: Path, failures: list[DownloadResult]) -> Path | None:
    if not failures:
        return None
    report_path = dataset_dir / "download-errors.csv"
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "image_id", "url", "error"])
        for failure in failures:
            writer.writerow(
                [failure.split, failure.image_id, failure.url, failure.error]
            )
    return report_path


def read_annotations_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ImageID": str, "LabelName": str})


def run(args: argparse.Namespace) -> RunSummary:
    dataset_dir, yaml_path, annotations_dir = prepare_output_paths(
        args.output_root, args.dataset, args.overwrite
    )
    csv_dir = args.csv_dir or annotations_dir
    ensure_metadata(csv_dir, args.splits, assume_yes=args.yes)
    copy_metadata(csv_dir, annotations_dir, args.splits)

    class_codes_by_name = resolve_class_codes(load_class_map(csv_dir), args.classes)
    class_ids = {
        code: index for index, code in enumerate(class_codes_by_name.values())
    }

    summary = RunSummary()
    failures: list[DownloadResult] = []
    for split in args.splits:
        annotations_path = csv_dir / ANNOTATION_FILES[split]
        annotations = read_annotations_csv(annotations_path)
        selected, image_ids = select_annotations(
            annotations, class_codes_by_name.values(), args.limit
        )
        labels_dir = dataset_dir / "labels" / split
        images_dir = dataset_dir / "images" / split
        summary.labels_written += write_yolo_labels(selected, class_ids, labels_dir)
        results = download_images(image_ids, split, images_dir, args.workers)
        remove_failed_labels(results, labels_dir)
        failures.extend(result for result in results if result.status == "failed")
        summary.downloaded_images += sum(
            1 for result in results if result.status == "downloaded"
        )
        summary.skipped_existing_images += sum(
            1 for result in results if result.status == "skipped"
        )
        summary.failed_downloads += sum(
            1 for result in results if result.status == "failed"
        )

    summary.labels_written = sum(1 for _ in (dataset_dir / "labels").glob("*/*.txt"))
    write_dataset_yaml(yaml_path, dataset_dir, args.classes)
    report_path = write_error_report(dataset_dir, failures)
    print_summary(summary, yaml_path, report_path)
    return summary


def print_summary(
    summary: RunSummary, yaml_path: Path, error_report: Path | None = None
) -> None:
    print(f"Downloaded images: {summary.downloaded_images}")
    print(f"Skipped existing images: {summary.skipped_existing_images}")
    print(f"Failed downloads: {summary.failed_downloads}")
    print(f"Labels written: {summary.labels_written}")
    print(f"Dataset YAML: {yaml_path}")
    if error_report is not None:
        print(f"Download errors: {error_report}")


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (FileExistsError, RuntimeError, ValueError, requests.RequestException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
