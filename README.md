# YOLOv5 RKNN

This directory contains step-by-step instructions for Transfer Learning using YOLOv5 pretrained weights. Starting from the YOLOv5 pretrained weights, we train a special-purpose classifier. The output is in PyTorch (`.pt`) format. We then convert this to RKNN format, to run natively on Rockchip NPU devices.

> [!NOTE]
> See my blog post [Edge AI using the Rockchip NPU](https://tristanpenman.com/blog/posts/2025/07/20/edge-ai-using-the-rockchip-npu/) for a high level overview of how all of this works. This includes technical information about the Rockchip RK3588 platform.
>
> The instructions here refer to the [Khadas Edge2](https://www.khadas.com/edge2) specifically, but should all be adaptable to other Rockchip devices.

### Contents

* [Background](#background)
  * [Transfer Learning](#transfer-learning)
  * [Quantization](#quantization)
  * [Calibration](#calibration)
* [Scripts](#scripts)
  * [RKNN Model Zoo](#rknn-model-zoo)
* [Prerequisites](#prerequisites)
  * [Dependencies](#dependencies)
  * [Models (optional)](#models-optional)
* [Conversion](#conversion)
  * [Testing](#testing)
* [Datasets](#datasets)
  * [YOLO Annotation Format](#yolo-annotation-format)
  * [Config File](#config-file)
* [Training](#training)

## Background

A few concepts to get out of the way...

### Transfer Learning

Transfer Learning refers to the practice of adapting a pretrained model - typically trained on a large dataset like ImageNet or COCO - for a different but related task.

The two main approaches to Transfer Learning are:

* **Feature Extraction**. We freeze the pretrained layers and train a new classifier (or head) for a new task.
* **Fine-Tuning**. In this case, we unfreeze some (or all) of the pretrained layers and train them along with the new layers on a custom dataset.

Feature extraction is well suited to cases where the target task is similar to the original task, or when the target domain is a subset of the original domain.

Fine-Tuning is more effective when the target dataset is large enough or differs significantly from the original domain. In practice, these two approaches are often combined: training begins with pure Feature Extraction, using frozen pretrained weights. Once the model achieves a reasonable level of performance, the pretrained weights are unfrozen, allowing them to adapt to the new task.

### Quantization

One of our goals in converting model weights to RKNN format is to reduce memory overhead through quantization. For example, a model trained with 32-bit floating point (FP32) weights can be quantised to 16-bit floating point (FP16), or even 8-bit integers (INT8), significantly reducing model size and improving inference efficiency, particularly on edge devices.

This process is well described in [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization).

### Calibration

Quantization relies on a selection of sample inputs, which are used to tune scale and zero point values for model weights.

In the case of YOLOv5, there is also a small change made to the model graph, whereby a post-processing step is moved outside the main model. This is to address stability issues introduced by quantization.

## Scripts

The conversion process is implemented in Python. The code for this lives in the [rknn](./rknn) directory. This includes scripts for converting the model on a Linux PC, as well as code for performing inference on a Rockchip device (e.g. RK3588). Inference can also be performed on a desktop CPU or GPU using ONNX.

The complete list of scripts is as follows:

* `coco_utils.py` - Code for working with the [COCO dataset](https://cocodataset.org).
* `convert.py` - Main conversion script. Uses RKNN-Toolkit2 to handle model conversion.
* `onnx_executor.py` - Classes that implement inference for arbitrary devices, as supported by [ONNX Runtime](https://github.com/microsoft/onnxruntime).
* `rknn_executor.py` - Classes that implement inference on Rockchip devices. Uses to RKNN-Toolkit2 to perform inference.
* `yolov5.py` - Wrapper script for performing inference. Relies on `ONNXExecutor` or `RKNNExecutor` to do the actual work.

### RKNN Model Zoo

The Python scripts in this repo are based on the `yolov5` example from [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo).

RKNN Model Zoo is a collection of examples that implement popular models using [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2).

## Prerequisites

Begin by cloning the repo and its submodules:

```
git clone git@github.com:tristanpenman/yolov5-rknn.git
cd yolov5-rknn
git submodule update --init yolov5
```

### Dependencies

Install requirements using `pip`:

```
pip3 install -r requirements.txt
```

This will install dependencies for the scripts in [rknn](./rknn) and [yolov5](./yolov5/).

### Models (optional)

You will also need to download any pretrained models that you want to use. These live in [models](./models). This repo includes `yolov5n.onnx` for convenience, so you only need to do this if you want to adapt another pretrained model.

## Conversion

We'll begin by converting and quantising a pretrained model to RKNN format. The relevant calibration dataset (`coco_subset_20`) and source model (`yolov5n.onnx`) have been included for convenience.

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

### Testing

At this stage, we should run the model on a Rockchip device to verify that it works.

You can follow the instructions in Khadas' [edge2-npu](https://github.com/khadas/edge2-npu/tree/master/C%2B%2B/yolov5) repo, or Rockchip's [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov5) repo.

## Datasets

Our goal is to construct a dataset that can be used to train a model for our own object classification tasks. The most straightforward way to do this is to find a dataset that is in YOLO Annotation Format already.

Alternatively, we can construct one ourselves!

### YOLO Annotation Format

The YOLOv5 annotation format is relatively simple. It assumes that each image has a corresponding .txt file with one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

There are no headers or image paths inside the .txt files. Each class is referred to by a number (i.e. its class ID). All coordinates are normalized to range between 0.0 and 1.0.

Finally, each annotation file has the same base name as the image file, e.g.:
```
image001.jpg
image001.txt
image002.jpg
image002.txt
...
```

### Config File

Write a YAML file like custom.yaml:

```
train: /path/to/dataset/images/train
val: /path/to/dataset/images/val

nc: 3  # number of classes
names: ['class_a', 'class_b', 'class_c']
```

## Training

python train.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --data custom.yaml \
  --cfg models/yolov5s.yaml \
  --weights yolov5s.pt \
  --name custom_yolov5s
