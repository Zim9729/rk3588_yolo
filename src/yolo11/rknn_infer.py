from pathlib import Path
from typing import Optional, Union

import numpy as np


class RknnLiteDetector:
    def __init__(self, model_path: Union[Path, str], core_mask: Optional[int] = None):
        self.model_path = Path(model_path)
        self.core_mask = core_mask
        self.rknn = None

    def load(self) -> "RknnLiteDetector":
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")

        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError(
                "rknn-toolkit-lite2 is required on the RK3588 board. "
                "Install the Rockchip wheel that matches your Python and board runtime."
            ) from exc

        rknn = RKNNLite()
        ret = rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_rknn failed with code: {ret}")

        if self.core_mask is None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(core_mask=self.core_mask)
        if ret != 0:
            rknn.release()
            raise RuntimeError(f"RKNN init_runtime failed with code: {ret}")

        self.rknn = rknn
        return self

    def infer(self, input_tensor: np.ndarray):
        if self.rknn is None:
            raise RuntimeError("RKNN runtime is not loaded")
        return self.rknn.inference(inputs=[input_tensor])

    def release(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def __enter__(self) -> "RknnLiteDetector":
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
