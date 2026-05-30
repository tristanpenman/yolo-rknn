# Downloader

[Back to README](../README.md)

The current custom dataset workflow depends on `OIDv6_ToolKit` to download Open Images data, then uses a second conversion step to turn those labels into YOLO format, followed by a shell script to reshape the files into the directory layout expected by YOLOv5.

This works, but it makes dataset preparation more complicated than it needs to be. The goal of the new downloader is to replace that workflow with a single `yolo_rknn.download` module that downloads Open Images examples and writes YOLO annotations directly.

## Goals

The downloader should:

* Download Open Images metadata CSV files when they are missing.
* Select images for one or more requested classes.
* Download the selected JPEG files.
* Write YOLO annotation files directly.
* Produce a YOLOv5-compatible dataset directory.
* Write the corresponding dataset YAML file.
* Support interrupted runs by skipping files that already exist.

The downloader should not rely on `OIDv6_ToolKit`, should not shell out to the AWS CLI, and should not write the intermediate OID label format currently converted by `annotations.py`.

## Proposed Command

The new module should be runnable with `python -m`:

```bash
python -m yolo_rknn.download apples-oranges \
  --classes Apple Orange \
  --splits train validation test \
  --limit 500 \
  --workers 20
```

For convenience, `--splits all` can expand to `train`, `validation`, and `test`:

```bash
python -m yolo_rknn.download apples-oranges \
  --classes Apple Orange \
  --splits all
```

The first positional argument is the dataset name. The selected classes determine the YOLO class IDs in the order provided on the command line.

## Output Layout

The downloader should write the same final layout currently produced by `scripts/prepare-dataset.sh`:

```text
datasets/
  apples-oranges/
    annotations/
      class-descriptions-boxable.csv
      train-annotations-bbox.csv
      validation-annotations-bbox.csv
      test-annotations-bbox.csv
    images/
      train/
      validation/
      test/
    labels/
      train/
      validation/
      test/
  apples-oranges.yaml
```

Each image should be written to:

```text
datasets/<dataset>/images/<split>/<ImageID>.jpg
```

Each label file should be written to:

```text
datasets/<dataset>/labels/<split>/<ImageID>.txt
```

Each label file should contain one row per object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Open Images bounding boxes are already normalized, so the conversion can be performed directly from the CSV values:

```text
x_center = (XMin + XMax) / 2
y_center = (YMin + YMax) / 2
width = XMax - XMin
height = YMax - YMin
```

This means the downloader does not need to read the image with OpenCV in order to convert labels.

## Dataset YAML

The generated YAML file should match the format used elsewhere in this repo:

```yaml
path: ../datasets/apples-oranges
train: images/train
test: images/test
val: images/validation

names:
  0: Apple
  1: Orange
```

The `path` value should remain relative to the `yolov5` submodule, because training commands are run through the bundled YOLOv5 checkout.

## Module Design

The implementation should live in:

```text
python/yolo_rknn/download.py
```

The module should be split into small functions that can be tested without network access.

### Argument Parsing

`parse_args()` should handle:

* `dataset`: dataset name, for example `apples-oranges`.
* `--classes`: one or more Open Images class names.
* `--splits`: one or more of `train`, `validation`, `test`, or `all`.
* `--output-root`: default `datasets`.
* `--csv-dir`: optional metadata cache directory.
* `--limit`: optional maximum number of images per split.
* `--workers`: concurrent image download count.
* `--overwrite`: allow replacing an existing dataset directory.
* `--yes`: allow non-interactive metadata downloads.

The first implementation can keep the interface close to the existing README examples. Additional Open Images filters can be added once the direct path is working.

### Metadata Downloads

`ensure_metadata(csv_dir, splits)` should download the required CSV files if they are missing:

* `class-descriptions-boxable.csv`
* `train-annotations-bbox.csv`
* `validation-annotations-bbox.csv`
* `test-annotations-bbox.csv`

The URL rules can follow the existing toolkit behavior:

* class descriptions from `https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv`
* train annotations from `https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv`
* validation and test annotations from `https://storage.googleapis.com/openimages/v5/`

The downloaded CSVs should also be copied into `datasets/<dataset>/annotations` so the generated dataset remains reproducible.

### Class Mapping

`load_class_map(csv_dir)` should read `class-descriptions-boxable.csv` and return a mapping from human-readable class names to Open Images label IDs.

For example:

```text
Apple -> /m/014j1m
Orange -> /m/0cyhj_
```

The downloader should fail clearly if a requested class name cannot be found.

### Annotation Selection

`select_annotations(df, class_codes, limit)` should:

* Filter rows to the selected Open Images class IDs.
* Preserve annotations for all selected classes in the same image.
* Apply the optional image limit after grouping by `ImageID`.
* Return both the selected annotation rows and the selected image IDs.

This differs from the current class-by-class directory flow. YOLO expects all labels for an image to live in one file, so multi-class images should produce one shared label file.

### YOLO Label Writing

`write_yolo_labels(annotations, class_ids, labels_dir)` should group rows by `ImageID` and write one `.txt` file per image.

The selected class order should define YOLO class IDs:

```text
Apple -> 0
Orange -> 1
```

Rows should be written with enough precision for normalized coordinates, for example:

```text
0 0.412345 0.533210 0.180000 0.270000
```

Empty label files should not be written unless a future training workflow explicitly needs background-only images.

### Image Downloads

`download_images(image_ids, split, images_dir, workers)` should download images directly over HTTP using `requests`.

The Open Images S3 URL pattern is:

```text
https://open-images-dataset.s3.amazonaws.com/<split>/<ImageID>.jpg
```

The function should:

* Skip images that already exist.
* Download concurrently with a bounded worker pool.
* Write to a temporary file before renaming into place.
* Record failed downloads and continue with the rest of the dataset.
* Remove labels for images that failed to download, so the dataset remains consistent.

## Error Handling

The downloader should avoid silent failures. At the end of a run it should print a short summary:

```text
Downloaded images: 487
Skipped existing images: 13
Failed downloads: 2
Labels written: 500
Dataset YAML: datasets/apples-oranges.yaml
```

If any downloads fail, write a small report:

```text
datasets/<dataset>/download-errors.csv
```

That file should include at least:

* split
* image ID
* URL
* error message

## Testing Plan

The first tests should not depend on network access. They can use small fixture CSVs and temporary directories.

Useful test cases:

* class names are mapped to Open Images label IDs correctly.
* unknown class names produce a clear error.
* Open Images boxes are converted directly to YOLO format.
* annotations for multiple selected classes in one image are written to one label file.
* `--limit` limits selected images rather than individual annotation rows.
* dataset directories and YAML files are created correctly.
* existing images are skipped during resume.

Network behavior can be tested separately by mocking the HTTP download function.

## Migration Plan

The migration can happen in small steps:

1. Add `python/yolo_rknn/download.py`.
2. Add unit tests for class mapping, annotation selection, YOLO conversion, and YAML generation.
3. Update the README's custom dataset section to use `python -m yolo_rknn.download`.
4. Keep `annotations.py` temporarily for anyone still using the old `OIDv6_ToolKit` flow.
5. Remove the `OIDv6_ToolKit` submodule and `scripts/prepare-dataset.sh` once the new downloader has replaced the old workflow.

The end state should be a simpler custom dataset path:

```bash
python -m yolo_rknn.download apples-oranges --classes Apple Orange --splits all
python yolov5/train.py --data datasets/apples-oranges.yaml
```
