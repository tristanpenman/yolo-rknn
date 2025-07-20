# YOLOv5 Experiments

A repo tracking experiments in fine-tuning YOLOv5 and converting model weights to RKNN format.

This is based on `yolov5` example from [rknn\_model\_zoo](https://github.com/airockchip/rknn_model_zoo). The RKNN Model Zoo relies on [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) for model conversion and quantisation.

## Scripts

The conversion process is implemented in Python. The code for this lives in the [rknn](./rknn) directory. This includes scripts for converting the model on a Linux PC. It also includes code for performing inference on a Rockchip device (e.g. RK3588) or on a desktop GPU (using ONNX).

The complete list of scripts is as follows:

* `coco_utils.py` - Code for working with the [COCO dataset](https://cocodataset.org).
* `convert.py` - Main conversion script. Uses RKNN-Toolkit2 to handle model conversion.
* `onnx_executor.py` - Classes that implement inference for arbitrary devices, as supported by [onnxruntime](https://github.com/microsoft/onnxruntime).
* `rknn_executor.py` - Classes that implement inference on Rockchip devices. Uses to RKNN-Toolkit2 to perform inference.
* `yolov5.py` - Wrapper script for performing inference. Relies on `ONNXExecutor` or `RKNNExecutor` to do the actual work.

### Quantisation

One of the goals of converting model weights to RKNN format is to reduce memory overhead through quantisation. This process is well described in [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization).

TODO: Describe the quantisation process.

### Calibration

Quantisation relies on a selection of sample inputs, which are used to tune scale and zero point values for model weights.

### Graph Alterations

In the case of YOLOv5, there is also a small change made to the model graph, whereby a post-processing step is moved outside the main model. This is to address stability issues introduced by quantisation.

TODO: Describe in more detail.

### Inference

Inference is pretty straightforward, thanks to being well abstracted by RKNN Toolkit and ONNX.

### Accuracy

TODO: Describe how we can evaluate accuracy.
