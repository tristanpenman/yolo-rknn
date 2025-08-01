# YOLOv5 RKNN

Experiments in fine-tuning YOLOv5 and converting model weights to RKNN format.

The Python code included in this repo is based on Rockchip's `yolov5` example from [rknn\_model\_zoo](https://github.com/airockchip/rknn_model_zoo). The RKNN Model Zoo uses [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) to perform model conversion and quantisation.

This repo also includes [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) as a submodule. This is used to perform Transfer Learning on a YOLOv5 pretrained model, and convert the model to ONNX format.

**Contents**

* [Scripts](#scripts)
* [Prerequisites](#prerequisites)
* [Background](#background)

More to come...

## Scripts

The conversion process is implemented in Python. The code for this lives in the [rknn](./rknn) directory. This includes scripts for converting the model on a Linux PC. It also includes code for performing inference on a Rockchip device (e.g. RK3588) or on a desktop GPU (using ONNX).

The complete list of scripts is as follows:

* `coco_utils.py` - Code for working with the [COCO dataset](https://cocodataset.org).
* `convert.py` - Main conversion script. Uses RKNN-Toolkit2 to handle model conversion.
* `onnx_executor.py` - Classes that implement inference for arbitrary devices, as supported by [onnxruntime](https://github.com/microsoft/onnxruntime).
* `rknn_executor.py` - Classes that implement inference on Rockchip devices. Uses to RKNN-Toolkit2 to perform inference.
* `yolov5.py` - Wrapper script for performing inference. Relies on `ONNXExecutor` or `RKNNExecutor` to do the actual work.

## Prerequisites

Begin by checking out this repo, including submodules:

```
git clone git@github.com:tristanpenman/yolov5-experiments.git
cd yolov5-experiments
git submodule update --init yolov5
```

Install requirements using `pip`:

```
pip3 install -r requirements.txt
```

## Background

A few concepts to get out of the way...

### Transfer Learning

Transfer Learning refers to the practice of using using a pretrained model (usually trained on a large dataset like ImageNet or COCO) for a different but related task.

The two main approaches to Transfer Learning are:

* **Feature Extraction**. We freeze the pretrained layers and train a new classifier (or head) for a new task.
* **Fine-Tuning**. In this case, we unfreeze some (or all) of the pretrained layers and train them along with the new layers on a custom dataset.

Feature extraction is well suited to cases where the target task is similar to the original task, or when the target domain is a subset of the original domain.

Fine-Tuning is more effective when the target dataset is large enough or differs significantly from the original domain. In practice, these two approaches are often combined: training begins with pure Feature Extraction, using frozen pretrained weights. Once the model achieves a reasonable level of performance, the pretrained weights are unfrozen, allowing them to adapt to the new task.

### Quantisation

One of the goals of converting model weights to RKNN format is to reduce memory overhead through quantisation. For example, a model trained with 32-bit floating point (FP32) weights can be quantised to 16-bit floating point (FP16), or even 8-bit integers (INT8), significantly reducing model size and improving inference efficiency, particularly on edge devices.

This process is well described in [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization).

### Calibration

Quantisation relies on a selection of sample inputs, which are used to tune scale and zero point values for model weights.

### Graph Alterations

In the case of YOLOv5, there is also a small change made to the model graph, whereby a post-processing step is moved outside the main model. This is to address stability issues introduced by quantisation.
