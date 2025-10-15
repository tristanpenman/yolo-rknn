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
  * [Apples and Oranges](#apples-and-oranges)
  * [Converting Annotations](#converting-annotations)
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

If you are using Linux or WSL, you should be able to proceed with [Direct Installation](#direct-installation). Docker is an option too, of course.

### Direct Installation

Install requirements using `pip`. Note that these instructions may depend on whether you're using pyenv or some other Python virtual environment:

```
pip3 install -r requirements.txt
```

This will install dependencies for the scripts in [scripts](scripts) and [yolov5](yolov5).

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

You will also need to download any pretrained models that you want to use. These live in [models](models). This repo includes `yolov5n.onnx` for convenience, so you only need to do this if you want to adapt another pretrained model.

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

Our goal now is to construct a custom dataset that can be used to train a model for our own object detection task. The most straightforward way to do this is to find a dataset that is in YOLO Annotation Format already.

Alternatively, we can construct one ourselves! To do this, we need to prepare a set of example images and object annotations in a YOLO-compatible format. The high level steps are described below.

### YOLO Annotation Format

The YOLOv5 annotation format is relatively simple. It assumes that each image has a corresponding .txt file, containing the 'labelled' data for that image. This is basically one line per object that has been identified in the image:
```
<class_id> <x_center> <y_center> <width> <height>
```

There are no headers or image paths inside these annotation .txt files. Each class is referred to by an integer (i.e. its class ID), and all coordinates are normalized to range between 0.0 and 1.0. Normalisation is illustrated below:

```
  |----- 300 px ----|

  0                 1
0 |------------------          -
  |                 |          |
  |                 |          |
  | |---------|     |          |
  | | 0.6x0.3 |-----|-- y=0.5  | 400 px
  | |---------|     |          |
  |      |          |          |
  |      |          |          |
1 |------------------          -
         |
       X=0.35
```

Finally, each annotation file has the same base name as the image file, e.g.:
```
# images
image001.jpg
image002.jpg

# labels
image001.txt
image002.txt
...
```

### Directory Structure

The dataset must follow a particular directory structure too:

```
└── datasets
    ├── firearms
    │   ├── images
    |   |   ├── test
    |   |   ├── train
    |   |   └── validation
    │   └── labels
    |       ├── test
    |       ├── train
    |       └── validation
    └── firearms.yaml
```

This example shows a dataset called `firearms`. Within the `firearms` directory, there are subdirectories for `images` and `labels`. These are, in turn, divided into the subdirectories `train`, `val`, and `test`. These are the training, validation and test splits of the data, respectively.

At the top level, there is also a `firearms.yaml` file that corresponds to the `firearms` directory.

### YAML File

The YAML file for a dataset contains paths to the training data, and a mapping from class IDs to class names:

```
path: ../datasets/firearms

train: images/train
test: images/test
val: images/val

names:
  0: Rifle
  1: Shotgun
  2: Handgun
```

Note: Due to the layout of this repo, the main `path:` is relative to the `yolov5` submodule. The `train`, `test` and `val` paths are relative to the main path.

### Apples and Oranges

Lets build our own dataset. Begin by downloading the images and annotations for an "Apples and Oranges" dataset, using `OIDv6_Toolkit`:

```bash
python OIDv6_ToolKit/main.py downloader --classes Apple Orange --type_csv all
```

This will prompt you to download various missing files:

* `class-descriptions-boxable.csv`
* `train-annotations-bbox.csv`
* `validation-annotations-bbox.csv`
* `test-annotations-bbox.csv`

Alternatively, the prompt can be suppressed by adding the `-y` option, ensuring that all files are downloaded automatically:

```bash
python OIDv6_ToolKit/main.py downloader -y --classes Apple Orange --type_csv all
```

The download may take a while, due to the number of images. The images will be downloaded to a directory called `OID`. Once the download is complete, we can inspect the structure of that directory using `tree`:

```
$ tree -d OID
.
├── Dataset
│   ├── test
│   │   ├── Apple
│   │   │   └── Label
│   │   └── Orange
│   │       └── Label
│   ├── train
│   │   ├── Apple
│   │   │   └── Label
│   │   └── Orange
│   │       └── Label
│   └── validation
│       ├── Apple
│       │   └── Label
│       └── Orange
│           └── Label
└── csv_folder

8 directories
```

This isn't quite what we need for YOLO, so lets proceed to converting annotations and preparing the dataset.

### Converting Annotations

The OID dataset includes four CSV files, containing metadata that we need for training:

* The `class-descriptions-boxable.csv` file is used when converting data into YOLO format. It provides a mapping between class label IDs and their human-readable names.
* The other three files (`train-annotations-bbox.csv`, `test-annotations-bbox.csv` and `validation-annotations-bbox.csv`) contain bounding box annotations for the images in each subset.

The next thing we need to do is convert the annotations to YOLO format. We have adapted the `convert_annotations.py` script from OIDv6_Toolkit, to make this more convenient:

```bash
./convert-annotations.py Apple Orange
```

This will create a new annotation file alongside each of the original JPEG files, showing progress meters

```
% ./convert-annotations.py Apple Orange
Currently in subdirectory: test
Converting annotations for class:  Apple
100%|█████████████████████████████████████████████████████| 144/144 [00:01<00:00, 109.78it/s]
Converting annotations for class:  Orange
100%|█████████████████████████████████████████████████████| 208/208 [00:03<00:00, 62.80it/s]
Currently in subdirectory: train
Converting annotations for class:  Apple
100%|█████████████████████████████████████████████████████| 1078/1078 [00:14<00:00, 76.54it/s]
Converting annotations for class:  Orange
100%|█████████████████████████████████████████████████████| 900/900 [00:20<00:00, 43.77it/s]
Currently in subdirectory: validation
Converting annotations for class:  Apple
100%|█████████████████████████████████████████████████████| 46/46 [00:00<00:00, 105.55it/s]
Converting annotations for class:  Orange
100%|█████████████████████████████████████████████████████| 61/61 [00:00<00:00, 108.31it/s]
```

Now we can move these into the datasets directory using the `prepare-dataset.sh` script:

```bash
./prepare-dataset.sh apples-oranges Apple Orange
```

The arguments are simply a dataset name (`apples-oranges`), then a list of the classes to be included (`Apple`, `Orange`).

This script will show progress while restructuring the data:

```
% ./prepare-dataset.sh apples-oranges Apple Orange
Preparing dataset directory: datasets/apples-oranges
Moving annotations...
Moving images and labels...
- subset train
  - class Apple
    - copying images
    - copying labels
  - class Orange
    - copying images
    - copying labels
- subset test
  - class Apple
    - copying images
    - copying labels
  - class Orange
    - copying images
    - copying labels
- subset validation
  - class Apple
    - copying images
    - copying labels
  - class Orange
    - copying images
    - copying labels
Writing yaml file...
Done!
```

The final dataset will be placed in `datasets/apples-oranges`. It will also create a YAML file (`datasets/apples-oranges.yaml`) that follows the same YAML file format described above.

We can inspect the final directory layout using `tree`:

```
% tree -d datasets
datasets
└── apples-oranges
    ├── annotations
    ├── images
    │   ├── test
    │   ├── train
    │   └── validation
    └── labels
        ├── test
        ├── train
        └── validation

11 directories
```

This looks just the `firearms` dataset we were using as a reference, and also includes an `annotations` directory. This contains the annotation CSV files we downloaded earlier:

* `class-descriptions-boxable.csv`
* `train-annotations-bbox.csv`
* `validation-annotations-bbox.csv`
* `test-annotations-bbox.csv`

## Ultralytics YOLOv5

With our custom dataset, we can now proceed to Feature Extraction, using the official Ultralytics distribution of YOLOv5. This makes it very easy to train a model on custom datasets.

The next few steps will tackle this at a high level. For more detail, the Ultralytics website has comprehensive documentation on how to [Train YOLOv5 on Custom Data](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/).

### Requirements

The Ultralytics `yolov5` repo has been included as a submodule. Its Python dependencies are included in the main `requirements.txt` file, so they should be installed already if you followed the [Prerequisites](#prerequisites) section.

If in doubt, you can re-run `pip` at the top-level of this repo:
```
pip install -r requirements.txt
```

### COCO128

By default, Ultralytics YOLOv5 will use a subset of COCO (specifically, COCO128) to train the model. Training with COCO128 can be started simply by running `train.py`:

```
python3 yolov5/train.py
```

**Note**: This will automatically download the COCO128 dataset to the [datasets](./datasets) directory (relative to the current directory when the script is executed).

This will generate A LOT of output from the training process. But what we care about are the results at the end of training:

```
Validating yolov5/runs/train/exp1/weights/best.pt...
Fusing layers...
Model summary: 157 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
                 Class     Images  Instances          P          R      mAP5    mAP50-95
                   all        128        929      0.921      0.906      0.963      0.784
                person        128        254      0.989      0.878      0.966      0.776
               bicycle        128          6      0.988          1      0.995      0.688
                   car        128         46          1      0.663      0.778      0.442
            motorcycle        128          5      0.935          1      0.995      0.932
              airplane        128          6      0.941          1      0.995      0.919
                   bus        128          7      0.923          1      0.995      0.863
                 train        128          3       0.91          1      0.995      0.895
                 truck        128         12      0.971      0.833      0.977       0.74
                  boat        128          6      0.727      0.893      0.955      0.659
         traffic light        128         14          1      0.616      0.811      0.427
             stop sign        128          2       0.88          1      0.995      0.895
                 bench        128          9          1      0.954      0.995       0.69
                  bird        128         16      0.978          1      0.995      0.846
                   cat        128          4          1      0.919      0.995      0.922
                   dog        128          9       0.97          1      0.995      0.878
                 horse        128          2      0.845          1      0.995      0.895
              elephant        128         17      0.979      0.941      0.967      0.846
                  bear        128          1       0.79          1      0.995      0.995
                 zebra        128          4      0.914          1      0.995      0.995
               giraffe        128          9      0.973          1      0.995       0.89
              backpack        128          6          1      0.931      0.995      0.712
              umbrella        128         18      0.982      0.944      0.961      0.785
               handbag        128         19      0.971      0.842      0.886      0.627
                   tie        128          7      0.961      0.857      0.858      0.694
              suitcase        128          4      0.932          1      0.995      0.865
               frisbee        128          5          1      0.932      0.995      0.785
                  skis        128          1      0.835          1      0.995      0.697
             snowboard        128          7      0.989      0.857       0.93      0.693
           sports ball        128          6          1      0.788      0.837      0.469
                  kite        128         10      0.967          1      0.995      0.638
          baseball bat        128          4          1      0.682      0.995       0.66
        baseball glove        128          7      0.789      0.571      0.668      0.446
            skateboard        128          5      0.937          1      0.995      0.832
         tennis racket        128          7      0.723      0.751      0.823      0.584
                bottle        128         18      0.932       0.76      0.951      0.642
            wine glass        128         16      0.854      0.732      0.928      0.636
                   cup        128         36      0.938      0.972      0.972      0.765
                  fork        128          6          1      0.906      0.995      0.862
                 knife        128         16          1      0.841       0.96       0.64
                 spoon        128         22      0.949      0.909      0.983      0.675
                  bowl        128         28      0.908      0.857      0.886      0.747
                banana        128          1      0.846          1      0.995      0.995
              sandwich        128          2      0.649          1      0.995      0.945
                orange        128          4      0.904          1      0.995      0.852
              broccoli        128         11      0.963          1      0.995       0.78
                carrot        128         24      0.919      0.948      0.989      0.779
               hot dog        128          2      0.858          1      0.995      0.895
                 pizza        128          5      0.951          1      0.995      0.879
                 donut        128         14      0.971          1      0.995       0.91
                  cake        128          4      0.912          1      0.995      0.964
                 chair        128         35      0.943      0.947      0.987      0.781
                 couch        128          6      0.903          1      0.995      0.979
          potted plant        128         14      0.974          1      0.995      0.887
                   bed        128          3      0.901          1      0.995      0.995
          dining table        128         13          1      0.856      0.982       0.82
                toilet        128          2      0.868          1      0.995      0.995
                    tv        128          2      0.855          1      0.995      0.895
                laptop        128          3      0.755      0.667      0.913      0.682
                 mouse        128          2       0.88          1      0.995      0.597
                remote        128          8      0.754       0.75      0.794      0.636
            cell phone        128          8      0.907      0.875      0.982      0.681
             microwave        128          3      0.899          1      0.995      0.963
                  oven        128          5       0.94          1      0.995      0.917
                  sink        128          6      0.912      0.833      0.922      0.762
          refrigerator        128          5      0.932          1      0.995      0.891
                  book        128         29      0.949      0.647      0.896      0.592
                 clock        128          9      0.967          1      0.995      0.866
                  vase        128          2      0.861          1      0.995      0.895
              scissors        128          1          1          0      0.995      0.464
            teddy bear        128         21       0.97          1      0.995       0.87
            toothbrush        128          5      0.894          1      0.995       0.86
Results saved to yolov5/runs/train/exp1
```

The key detail we need to extract from this is `yolov5/runs/train/exp1/weights/best.pt`, which is the path the weights at the end of training. These are stored in PyTorch format, hence the `.pt` extension.

Note also that the path includes `exp1`, which is the current 'experiment' number. This will be incremented each time you run `train.py`.

### Training

We can easily adapt this for other datasets, such as those that we prepared earlier (e.g. Apples and Oranges).

```
python yolov5/train.py \
  --img 640 \
  --batch 16 \
  --epochs 20 \
  --data datasets/apples-oranges.yaml
```

While this is running, you can see the progress of each epoch (slightly cleaned up for readability):

```
Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
12/19      3.69G     0.0392    0.05197   0.002492        112        640
           Class     Images  Instances          P          R      mAP50   mAP50-95
             all        107        277      0.423      0.545      0.433      0.341
```

This can be read as two pairs of rows. The first row of each pair contains attributes, and the second row contains values.

Some of the most interesting values here are the box loss, object loss and class loss scores. See the YOLO documentation to understand how these values should be interpreted.

We can also see the mAP scores for each iteration - these are described in the [Evaluation](#evaluation) section below.

Once training is complete, you'll see the final scores:

```
Validating yolov5/runs/train/exp2/weights/best.pt...
Fusing layers...
Model summary: 157 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all        107        277      0.473      0.512      0.469      0.374
                 Apple        107        102      0.652      0.578      0.613      0.489
                Orange        107        175      0.294      0.446      0.325       0.26
Results saved to yolov5/runs/train/exp2
```

This is much more compact than earlier, because there are only two classes to evaluate.

### Evaluation

TODO: Discuss mAP, etc

## Deployment


Once satisfied with the results, we can deploy the model to our Edge2 device. But in order to do this, we must convert the fine-tuned model to RKNN format, following the same steps covered in the [Conversion](#conversion) section above.

### ONNX Format

The first step is to convert the model to ONNX format:

```
python yolov5/export.py \
  --data datasets/apples-oranges.yaml \
  --include onnx \
  --weights yolov5/runs/train/exp1/weights/best.pt \
  --img 640
```

We use the `--data` option to specify the dataset we've used, which can be used to configure the number of outputs in the model. We provide the path to the latest model using `--weights`, and the image size using `--img`. Finally, we specify `--include onnx` to export to ONNX format.

Once the export is complete, you can see that the model has been saved as `yolov5/runs/train/exp1/weights/best.onnx`. Output from the conversion process should look similar to this:

```
Fusing layers...
Model summary: 157 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs

PyTorch: starting from yolov5/runs/train/exp1/weights/best.pt with output shape (1, 25200, 7) (13.7 MB)

ONNX: starting export with onnx 1.18.0...
ONNX: export success ✅ 0.8s, saved as yolov5/runs/train/exp1/weights/best.onnx (27.2 MB)

Export complete (1.1s)
Results saved to /home/tristan/Workspace/yolov5-rknn/yolov5/runs/train/exp1/weights
Detect:          python detect.py --weights yolov5/runs/train/exp1/weights/best.onnx
Validate:        python val.py --weights yolov5/runs/train/exp1/weights/best.onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/exp1/weights/best.onnx')
Visualize:       https://netron.app
```

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
