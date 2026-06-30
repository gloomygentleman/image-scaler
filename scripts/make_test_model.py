#!/usr/bin/env python3
"""Generate a tiny, deterministic ONNX super-resolution model + sample images.

This is a DEV/VERIFICATION tool only — it is not part of the shipped app.

The model is a single `Resize` node (nearest, integer scale) that conforms to the
exact I/O contract the Rust app expects:
    input : NCHW, RGB, float32 in [0, 1], dynamic H/W
    output: NCHW, RGB, float32 in [0, 1], H/W multiplied by `scale`

Because the transform is a plain nearest-neighbour upscale, the Rust pipeline's
output can be checked against a trivially-computed reference (see verify_output).
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
SAMPLES = os.path.join(ROOT, "samples")


def make_model(scale: int, path: str) -> None:
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, float(scale), float(scale)], dtype=np.float32),
        name="scales",
    )
    node = helper.make_node(
        "Resize",
        inputs=["input", "", "scales"],  # empty roi
        outputs=["output"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph(
        [node],
        "nearest_upscale",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "H", "W"])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "Hs", "Ws"])],
        initializer=[scales],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote model: {path}  (nearest x{scale})")


def make_sample_rgb(w: int, h: int, path: str) -> None:
    """Gradient background with sharp checker squares (edges to make SR visible)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    arr[..., 0] = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
    arr[..., 1] = (yy * 255 // max(h - 1, 1)).astype(np.uint8)
    arr[..., 2] = ((xx + yy) * 255 // max(w + h - 2, 1)).astype(np.uint8)
    checker = (((xx // 12) + (yy // 12)) % 2).astype(bool)
    arr[checker] = 255 - arr[checker]
    Image.fromarray(arr, "RGB").save(path)
    print(f"wrote sample: {path}  ({w}x{h} RGB)")


def make_sample_rgba(w: int, h: int, path: str) -> None:
    """RGB with a radial alpha falloff, to test transparency preservation."""
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    rgb = np.zeros((h, w, 4), dtype=np.uint8)
    rgb[..., 0] = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
    rgb[..., 1] = (yy * 255 // max(h - 1, 1)).astype(np.uint8)
    rgb[..., 2] = 128
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(255 - (dist / (max(w, h) / 2.0)) * 255, 0, 255).astype(np.uint8)
    rgb[..., 3] = alpha
    Image.fromarray(rgb, "RGBA").save(path)
    print(f"wrote sample: {path}  ({w}x{h} RGBA)")


def verify_output(input_path: str, output_path: str, scale: int) -> None:
    """Check the Rust output against a reference nearest-neighbour upscale."""
    src = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.int32)
    out = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.int32)
    h, w = src.shape[:2]
    exp_h, exp_w = h * scale, w * scale
    assert out.shape[:2] == (exp_h, exp_w), f"dims {out.shape[:2]} != {(exp_h, exp_w)}"
    ref = np.repeat(np.repeat(src, scale, axis=0), scale, axis=1)
    mad = float(np.mean(np.abs(out - ref)))
    print(f"output {out.shape[1]}x{out.shape[0]}  mean-abs-diff vs nearest-ref = {mad:.3f}")
    assert mad < 2.0, f"output differs too much from nearest reference (MAD={mad})"
    print("PASS: dimensions and pixels match nearest-upscale reference")


def main() -> None:
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(SAMPLES, exist_ok=True)

    if len(sys.argv) >= 2 and sys.argv[1] == "verify":
        # verify <input> <output> <scale>
        verify_output(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return

    make_model(2, os.path.join(MODELS, "test_resize_x2.onnx"))
    make_model(4, os.path.join(MODELS, "test_resize_x4.onnx"))
    make_sample_rgb(192, 144, os.path.join(SAMPLES, "sample_in.png"))
    make_sample_rgba(160, 120, os.path.join(SAMPLES, "sample_rgba.png"))


if __name__ == "__main__":
    main()
