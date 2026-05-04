"""HiFi-GAN Generator を CoreML (.mlpackage) に変換するスクリプト

通常変換:
    python scripts/convert_hifigan.py --precision float16

shape 違いの検証用バリアントを生成:
    python scripts/convert_hifigan.py --precision float32 --shape-mode range1
    python scripts/convert_hifigan.py --all-variants

`--shape-mode` を指定すると出力ファイル名は
`HiFiGAN_Generator_<precision>_<shape_mode>.mlpackage` となる。
未指定（legacy 動作）の場合は従来どおり `HiFiGAN_Generator_<precision>.mlpackage`。
"""

import argparse
import sys
import os
import json

import numpy as np
import torch
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.pardir)

# HiFiGAN モジュールのインポートパスを通す
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PronounSE", "HiFiGAN"))
from models import Generator

HIFIGAN_CHKPT_DIR = os.path.join(PROJECT_ROOT, "PronounSE", "HiFiGAN", "chkpt")

# shape_mode → (RangeDim 設定 もしくは 固定 T, デフォルト T)
# default_t は trace 用ダミー入力と PyTorch / CoreML 出力比較に使うフレーム数
SHAPE_MODES = {
    "range1":       {"kind": "range", "lower": 1,  "upper": 1000, "default": 100, "trace_t": 100},
    "range16":      {"kind": "range", "lower": 16, "upper": 1000, "default": 100, "trace_t": 100},
    "range16_384":  {"kind": "range", "lower": 16, "upper": 384,  "default": 262, "trace_t": 262},
    "fixed262":     {"kind": "fixed", "size": 262, "trace_t": 262},
}

# --all-variants で生成する組み合わせ
ALL_VARIANTS = [
    ("float32", "range1"),
    ("float32", "range16"),
    ("float32", "range16_384"),
    ("float32", "fixed262"),
    ("float16", "range16"),
    ("float16", "range16_384"),
    ("float16", "fixed262"),
]


def get_output_path(precision, shape_mode=None):
    if shape_mode is None:
        # legacy naming（既存アプリが参照）
        return os.path.join(PROJECT_ROOT, f"HiFiGAN_Generator_{precision}.mlpackage")
    return os.path.join(
        PROJECT_ROOT, f"HiFiGAN_Generator_{precision}_{shape_mode}.mlpackage"
    )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_generator():
    """Generator を CPU でロードし weight_norm を除去して返す"""
    with open(os.path.join(HIFIGAN_CHKPT_DIR, "config.json")) as f:
        config = json.load(f)
    param = AttrDict(config)

    generator = Generator(param).cpu()
    chkpt = torch.load(
        os.path.join(HIFIGAN_CHKPT_DIR, "g_00009000"),
        map_location="cpu",
    )
    generator.load_state_dict(chkpt["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator


def get_compute_precision(precision_str):
    if precision_str == "float16":
        return ct.precision.FLOAT16
    elif precision_str == "float32":
        return ct.precision.FLOAT32
    else:
        return ct.precision.FLOAT16


def quantize_int8(mlmodel):
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config = OptimizationConfig(global_config=op_config)
    return linear_quantize_weights(mlmodel, config=config)


def build_input_shape(shape_mode):
    """shape_mode に応じた ct.TensorType の shape を返す"""
    if shape_mode is None:
        # legacy: RangeDim(16, 1000, default=100)
        return ct.Shape(shape=(1, 256, ct.RangeDim(16, 1000, default=100)))

    spec = SHAPE_MODES[shape_mode]
    if spec["kind"] == "range":
        return ct.Shape(shape=(
            1, 256,
            ct.RangeDim(spec["lower"], spec["upper"], default=spec["default"]),
        ))
    else:
        return (1, 256, spec["size"])


def trace_t_for(shape_mode):
    if shape_mode is None:
        return 100
    return SHAPE_MODES[shape_mode]["trace_t"]


def convert_one(generator, precision, shape_mode):
    """1 パターン分の .mlpackage を生成する"""
    label = f"{precision}" + (f" / {shape_mode}" if shape_mode else " (legacy)")
    print(f"\n=== 変換開始: {label} ===")

    trace_t = trace_t_for(shape_mode)
    dummy_input = torch.randn(1, 256, trace_t)

    with torch.no_grad():
        pt_output = generator(dummy_input).numpy()
    print(f"PyTorch 出力 shape: {pt_output.shape}")

    print("torch.jit.trace 実行中...")
    traced = torch.jit.trace(generator, dummy_input)

    print("CoreML 変換中...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=build_input_shape(shape_mode)),
        ],
        convert_to="mlprogram",
        compute_precision=get_compute_precision(precision),
        # 旧 opset (iOS15) のままだとシミュレータの MPSGraph で MLIR pass が
        # 落ちて Float32 + GPU が abort する。最新 opset で再変換する
        minimum_deployment_target=ct.target.iOS26,
    )

    if precision == "int8":
        print("Int8 量子化中...")
        mlmodel = quantize_int8(mlmodel)

    output_path = get_output_path(precision, shape_mode)
    mlmodel.save(output_path)
    print(f"保存完了: {output_path}")

    print("出力比較中...")
    prediction = mlmodel.predict({"mel": dummy_input.numpy()})
    output_key = list(prediction.keys())[0]
    coreml_output = prediction[output_key]

    max_abs_error = np.max(np.abs(pt_output - coreml_output))
    is_close = np.allclose(pt_output, coreml_output, atol=1e-4)
    print(f"最大絶対誤差: {max_abs_error:.6e}")
    print(f"np.allclose(atol=1e-4): {is_close}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        choices=["float16", "float32", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--shape-mode",
        choices=list(SHAPE_MODES.keys()),
        default=None,
        help="入力 shape のバリアント。省略時は legacy 動作 (RangeDim 16-1000)",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Float32/Float16 × shape 4 種の計 7 パターンを一括生成",
    )
    args = parser.parse_args()

    print("Generator をロード中...")
    generator = load_generator()

    if args.all_variants:
        print(f"--all-variants: {len(ALL_VARIANTS)} 通り生成します")
        for precision, shape_mode in ALL_VARIANTS:
            convert_one(generator, precision, shape_mode)
        print("\n=== すべての変換が完了 ===")
    else:
        convert_one(generator, args.precision, args.shape_mode)


if __name__ == "__main__":
    main()
