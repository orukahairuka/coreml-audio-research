"""HiFi-GAN Generator を CoreML (.mlpackage) に変換するスクリプト"""

import sys
import os
import json

import numpy as np
import torch
import coremltools as ct

# HiFiGAN モジュールのインポートパスを通す
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PronounSE", "HiFiGAN"))
from models import Generator

HIFIGAN_CHKPT_DIR = os.path.join(os.path.dirname(__file__), "PronounSE", "HiFiGAN", "chkpt")
OUTPUT_PATH = "HiFiGAN_Generator.mlpackage"


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


def main():
    # 1. モデルロード
    print("Generator をロード中...")
    generator = load_generator()

    # 2. ダミー入力
    dummy_input = torch.randn(1, 256, 100)

    # 3. PyTorch 出力（比較用）
    with torch.no_grad():
        pt_output = generator(dummy_input).numpy()
    print(f"PyTorch 出力 shape: {pt_output.shape}")

    # 4. TorchScript trace
    print("torch.jit.trace 実行中...")
    traced = torch.jit.trace(generator, dummy_input)

    # 5. CoreML 変換
    print("CoreML 変換中...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="mel",
                shape=ct.Shape(
                    shape=(1, 256, ct.RangeDim(1, 1000, default=100))
                ),
            )
        ],
        convert_to="mlprogram",
    )
    mlmodel.save(OUTPUT_PATH)
    print(f"保存完了: {OUTPUT_PATH}")

    # 6. 変換前後の出力比較
    print("出力比較中...")
    prediction = mlmodel.predict({"mel": dummy_input.numpy()})
    # 出力キー名を取得
    output_key = list(prediction.keys())[0]
    coreml_output = prediction[output_key]

    max_abs_error = np.max(np.abs(pt_output - coreml_output))
    is_close = np.allclose(pt_output, coreml_output, atol=1e-4)

    print(f"最大絶対誤差: {max_abs_error:.6e}")
    print(f"np.allclose(atol=1e-4): {is_close}")


if __name__ == "__main__":
    main()
