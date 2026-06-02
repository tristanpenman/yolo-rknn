"""RKNN model container for YOLOv5.

Adapted from RKNN Toolkit 2.

Originally released under the BSD 3-Clause "New" or "Revised" License.
"""

from rknn.api import RKNN


class RknnModelContainer:
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target is None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        self.rknn = rknn

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)

        return result

    def release(self):
        self.rknn.release()
        self.rknn = None
