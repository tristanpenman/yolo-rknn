# YOLOv5 RKNN

This repo contains step-by-step instructions to perform Transfer Learning using YOLOv5 pretrained weights. Starting from YOLOv5 pretrained weights, we train a special-purpose object detection model. Initially, the custom model will be stored in PyTorch (`.pt`) format. We then convert this to RKNN format, allowing it to run natively on a Rockchip RK3588 NPU.

> [!NOTE]
> See my blog post [Edge AI using the Rockchip NPU](https://tristanpenman.com/blog/posts/2025/07/20/edge-ai-using-the-rockchip-npu/) for a high level overview of how all of this works. This includes technical information about the Rockchip RK3588 platform.
>
> These instructions refer to the [Khadas Edge2](https://www.khadas.com/edge2) specifically, but it should be relatively easy to adapt this to other Rockchip NPU devices.

### Contents

* [Background](#background)
  * [Transfer Learning](#transfer-learning)
  * [Quantization](#quantization)
  * [Scaling](#scaling)
* [Prerequisites](#prerequisites)
  * [Direct Installation](#direct-installation)
  * [Docker (alternative)](#docker-alternative)
  * [Models (optional)](#models-optional)
* [Conversion](#conversion)
  * [Scripts](#scripts)
  * [RKNN Model Zoo](#rknn-model-zoo)
  * [ONNX to RKNN](#onnx-to-rknn)
  * [Verification](#verification)
* [Custom Dataset](#custom-dataset)
  * [YOLO Annotation Format](#yolo-annotation-format)
  * [Config File](#config-file)
  * [Cats and Dogs](#cats-and-dogs)
* [Ultralytics YOLOv5](#ultralytics-yolov5)
  * [Requirements](#Requirements)
  * [COCO128](#coco128)
  * [Training](#training)
  * [Evaluation](#evaluation)
* [Deployment](#deployment)
  * [ONNX Format](#onnx-format)
  * [Calibration](#calibration)
  * [RKNN Format](#rknn-format)

## Background

A few concepts to get out of the way...

### Transfer Learning

Transfer Learning refers to the practice of adapting a pretrained model (typically trained on a large computer vision dataset such as ImageNet or COCO) for a different but related task.

The two main approaches to Transfer Learning are:

* **Feature Extraction**. We freeze the pretrained layers and train a new classifier (or head) for a new task. Freezing a layer means that its weights won't change.
* **Fine-Tuning**. We unfreeze some (or all) of the pretrained layers, allowing the weights to change. Therefore, existing layers will also adapt to the new custom dataset.

There are cases where we might use one approach, or the other.

Feature extraction is well suited to cases where the target task is similar to the original task, or when the target domain is a subset of the original domain.

Fine-Tuning is more effective when the target dataset is large enough or differs significantly from the original domain. In practice, these two approaches are often combined: training begins with pure Feature Extraction, using frozen pretrained weights. Once the model achieves a reasonable level of performance, the pretrained weights are unfrozen, allowing them to adapt to the new task.

### Quantization

One of our goals in converting model weights to RKNN format is to reduce memory overhead through quantization. For example, a model trained with 32-bit floating point (FP32) weights can be quantised to 16-bit floating point (FP16), or even 8-bit integers (INT8), significantly reducing model size and improving inference efficiency on Edge devices.

This process is well described in [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization).

### Scaling

Quantization will cause model weights and activations to be scaling, and possibly shifted to a new zero point. These scale and zero point values are chosen (or calibrated) based on a subset of data.

In the case of YOLOv5, we also need to make a small change to the model graph, whereby a post-processing step is moved outside the main model. This is to address stability issues introduced by quantization.

## Prerequisites

Begin by cloning the repo and its submodules:

```
git clone git@github.com:tristanpenman/yolov5-rknn.git
cd yolov5-rknn
git submodule update --init yolov5
```

What you do next will depend on your operating system. If you are using macOS, you will need to use Docker as [described below](#docker-optional).

If you are using Linux or WSL, you should be able to proceed with Direct Installation. Docker is an option too, of course.

### Direct Installation

Install requirements using `pip`. Note that these instructions may depend on whether you're using pyenv or some other Python virtual environment:

```
pip3 install -r requirements.txt
```

This will install dependencies for the scripts in [scripts](scripts) and [yolov5](./yolov5/).

### Docker (optional)

Build the Docker image:

```
docker build -t yolov5-rknn .
```

Once the image has been built, you can start a container using:

```
docker run -it --rm -v "$PWD:/workspace" yolov5-rknn
```

### Models (optional)

You will also need to download any pretrained models that you want to use. These live in [models](./models). This repo includes `yolov5n.onnx` for convenience, so you only need to do this if you want to adapt another pretrained model.

## Conversion

The model conversion process is implemented in Python. The code for this lives in the [scripts](scripts) directory. This includes scripts for converting the model on a Linux PC, as well as code for performing inference on a Rockchip device (e.g. RK3588). Inference can also be performed on a desktop CPU or GPU using ONNX.

### Scripts

The complete list of scripts is as follows:

* `coco_utils.py` - Code for working with the [COCO dataset](https://cocodataset.org).
* `convert.py` - Main conversion script. Uses RKNN-Toolkit2 to handle model conversion.
* `onnx_executor.py` - Classes that implement inference for arbitrary devices, as supported by [ONNX Runtime](https://github.com/microsoft/onnxruntime).
* `rknn_executor.py` - Classes that implement inference on Rockchip devices. Uses to RKNN-Toolkit2 to perform inference.
* `yolov5.py` - Wrapper script for performing inference. Relies on `ONNXExecutor` or `RKNNExecutor` to do the actual work.

### RKNN Model Zoo

These scripts are based on the `yolov5` example from [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo). RKNN Model Zoo is a collection of examples that implement popular models using [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2).

These have been tidied up and simplified for inclusion here.

### ONNX to RKNN

We'll begin by converting and quantising a pretrained ONNX model to RKNN format. The relevant calibration dataset (`coco_subset_20`) and source model (`yolov5n.onnx`) have been included for convenience.

The conversion script (`convert.py`) takes the follow command line arguments:

    Usage: python3 convert.py <onnx_model_path> <dataset_path> <platform> [dtype(optional)] [output_rknn_path(optional)]
              platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1103, rv1106, rv1126b, rv1109, rv1126, rk1808]
              dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1103, rv1106, rv1126b]
              dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]

The `dataset_path` refers to a file containing a list of images to be used for calibration. We can use `coco_subset_20.txt` for now. This contains a subset of COCO images, making it appropriate for this task.

Convert the model by running the following:
```
python3 rknn/convert.py models/yolov5n.onnx coco_subset_20.txt rk3588
```

The output should look something like this (some detail omitted):
```
I rknn-toolkit2 version: 2.3.2
--> Config model
done
--> Loading model
I Loading : ...
done
--> Building model
I OpFusing 0 ...
I OpFusing 1 ...
I OpFusing 2 ...
W build: found outlier value, this may affect quantization accuracy
                        const name          abs_mean    abs_std     outlier value
                        onnx::Conv_347      0.68        0.89        -11.603
I GraphPreparing ...
I Quantizating ...
W build: The default input dtype of 'images' is changed from 'float32' to 'int8' in rknn model for performance!
                       Please take care of this change when deploy rknn model with Runtime API!
W build: The default output dtype of 'output0' is changed from 'float32' to 'int8' in rknn model for performance!
                      Please take care of this change when deploy rknn model with Runtime API!
I rknn building ...
I rknn building done.
done
--> Export rknn model
done
```

The converted model will be written to `yolov5.rknn`.

### Verification

At this stage, we should run the model on a Rockchip device to verify that it works.

You can follow the instructions in Khadas' [edge2-npu](https://github.com/khadas/edge2-npu/tree/master/C%2B%2B/yolov5) repo, or Rockchip's [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov5) repo.

## Custom Dataset

Our goal is to construct a custom dataset that can be used to train a model for our own object detection task. The most straightforward way to do this is to find a dataset that is in YOLO Annotation Format already.

Alternatively, we can construct one ourselves! To do this, we need to prepare a set of example images and object annotations in a YOLO-compatible format.

### YOLO Annotation Format

The YOLOv5 annotation format is relatively simple. It assumes that each image has a corresponding .txt file with one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

There are no headers or image paths inside the .txt files. Each class is referred to by an integer (i.e. its class ID). All coordinates are normalized to range between 0.0 and 1.0.

```
  0                 1
0 |------------------
  |                 |
  |                 |
  | |---------|     |
  | | 0.6x0.3 |-----|-- y=0.5
  | |---------|     |
  |      |          |
  |      |          |
1 |------------------
         |
       X=0.35
```

Finally, each annotation file has the same base name as the image file, e.g.:
```
image001.jpg
image001.txt
image002.jpg
image002.txt
...
```

### Directory Structure

Printing out the directory tree using `tree` should look something like this:

```
% tree . -L 3
.
├── datasets
│   ├── firearms
│   │   ├── images
│   │   └── labels
│   └── firearms.yaml
└── yolov5
    ├── benchmarks.py
    ├── CITATION.cff
    ├── classify
    |
    :   // trimmed for brevity //
    |
    ├── train.py
    └── val.py

22 directories, 61 files
```

### Config File

Write a YAML file like custom.yaml:

```
train: /path/to/dataset/images/train
val: /path/to/dataset/images/val

nc: 3  # number of classes
names: ['class_a', 'class_b', 'class_c']
```

### Cats and Dogs

TODO: Build the dataset

## Ultralytics YOLOv5

With our custom dataset, we can now proceed to Feature Extraction, using the official Ultralytics distribution of YOLOv5. This makes it very easy to train a model on custom datasets.

The next few steps will tackle this at a high level. For more detail, the Ultralytics website has comprehensive documentation on how to [Train YOLOv5 on Custom Data](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/).

### Requirements

The Ultralytics `yolov5` repo has been included as a submodule. Its Python dependencies are included in the main `requirements.txt` file, so they should be installed already.

If in doubt, you can re-run `pip` at the top-level of this repo:
```
pip install -r requirements.txt
```

### COCO128

By default, Ultralytics YOLOv5 will use a subset of COCO (specifically, COCO128) to train the model. Training with COCO128 can be started simply by running `train.py`:

```
python3 yolov5/train.py
```

**Note**: This will automatically download the COCO128 dataset to the [datasets](./datasets) directory. This directory is ignored by `.gitignore`, but you should still take care to _not_ commit any data files into the repo.

### Training

We can easily adapt this for other datasets, such as those that we prepared earlier (e.g. Cats and Dogs).

```
python yolov5/train.py \
  --img 640 \
  --batch 16 \
  --epochs 20 \
  --data datasets/cats-and-dogs.yaml
```

TODO: Output

You can see the progress of each epoch:

TODO: Output

This shows the box, object and class loss scores. It also shows mAP scores for each iteration. See the YOLO documentation to understand how these values should be interpreted.

Once training is complete, you'll see the final scores:

TODO: Output

### Evaluation

TODO: Discuss mAP, etc

## Deploymnet

Once satisfied with the results, we can deploy the model to our Edge2 device. First we need to convert the model to RKNN format, as described in the [Conversion](#conversion) section.

The model trained by Ultralytics YOLOv5 will be in PyTorch format. We need to figure out how to convert

### ONNX Format

The first step is to convert the model to ONNX format:

```
python yolov5/export.py \
  --data datasets/firearms.yaml \
  --include onnx \
  --weights yolov5/runs/train/exp12/weights/best.pt \
  --img 640
```

We use the `--data` option to specify the dataset we've used, which can be used to configure the number of outputs in the model. We provide the path to the latest model using `--weights`, and the image size using `--img`. Finally, we specify `--include onnx` to export to ONNX format.

TODO: Example

Once the export is complete, you can see that the model has been saved as `yolov5/runs/train/exp12/weights/best.onnx`.

Converting from ONNX to RKNN is a little more involved...

### Calibration

To convert to RKNN format, we need to choose a subset of the training data to use for quantization and calibration. Recall that calibration will scale the weights and activations of the model, to fit a smaller or less precise data type. This doesn't require a lot of data - just enough to produce reasonable values scale factors and zero points.

We can do this using just 10 examples.

What's the easiest way to copy 10 random files from a directory using standard Linux/Unix command line tools? We can combine `find`, `shuf` and `xargs`:

```
find /path/to/dir -type f | shuf -n 10 | xargs -I{} cp {} /destination/dir
```

For example, from the top-level `yolov5-transfer-learning` directory:

```
mkdir apples-oranges-calib
find datasets/apples-oranges -type f -name '*.jpg' | shuf -n 100 | xargs -I{} cp {} apples-oranges-calib
```

Then we can create a list of relative paths to the images in that directory:

```
find apples-oranges-calib -type f -exec echo "./{}" \; > apples-oranges-calib.txt
```

We should also check the output:

```
$ head apples-oranges-calib.txt
./apples-oranges-calib/69974d01b659acfe.jpg
./apples-oranges-calib/326cd1e8ed154d23.jpg
./apples-oranges-calib/6a7745c9f562a645.jpg
./apples-oranges-calib/7d5be279c905b3fa.jpg
./apples-oranges-calib/1bee85275cc7f4c8.jpg
./apples-oranges-calib/56499fcbf2b50447.jpg
./apples-oranges-calib/48a3376ae485a0d0.jpg
./apples-oranges-calib/4f8fc6120801f196.jpg
./apples-oranges-calib/777db5187bd548c0.jpg
./apples-oranges-calib/6dd64d2dee5b9920.jpg
```

### RKNN Format

We're finally ready to run the RKNN `convert.py` script using the new calibration dataset:

```
python3 rknn/convert.py \
  yolov5/runs/train/exp12/weights/best.onnx \
  apples-oranges-calib.txt \
  rk3588
```

The output should look like this:

TODO: include output
