"""Transformer (Encoder / Decoder) を CoreML (.mlpackage) に変換するスクリプト

推論時は Encoder → Decoder (自己回帰ループ) の順で呼ばれるため、
それぞれ別の CoreML モデルとして変換する。
"""

import argparse
import sys
import os

import numpy as np
import torch
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.pardir)

# Transformer モジュールのインポートパスを通す
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PronounSE", "Transformer"))
from network import Model
import hyperparams as hp

TRANSFORMER_CHKPT = os.path.join(
    PROJECT_ROOT, "PronounSE", "Transformer", "chkpt", "chkpt__20000.pth.tar"
)
ENCODER_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Transformer_Encoder.mlpackage")
DECODER_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Transformer_Decoder.mlpackage")


class EncoderWrapper(torch.nn.Module):
    """procEncoder のラッパー。CoreML 用に出力を memory のみに絞る。
    eval モードでは c_mask=None, attns は不要なため。"""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, mel, pos):
        memory, _, _ = self.encoder(mel, pos)
        return memory


class DecoderWrapper(torch.nn.Module):
    """procDecoder のラッパー。eval モードでは c_mask=None なので引数から除外。"""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, memory, decoder_input, pos):
        mel_out, postnet_out, _, _ = self.decoder(memory, decoder_input, None, pos)
        return mel_out, postnet_out


def load_model():
    """Transformer モデルを CPU でロードして eval モードにする"""
    model = Model(hp.prenet_type).cpu()
    chkpt = torch.load(TRANSFORMER_CHKPT, map_location="cpu")
    model.load_state_dict(chkpt["model"])
    model.eval()
    return model


def get_compute_precision(precision_str):
    if precision_str == "float16":
        return ct.precision.FLOAT16
    elif precision_str == "float32":
        return ct.precision.FLOAT32
    else:
        # int8: まずfloat16で変換し、後で量子化する
        return ct.precision.FLOAT16


def quantize_int8(mlmodel):
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config = OptimizationConfig(global_config=op_config)
    return linear_quantize_weights(mlmodel, config=config)


def convert_encoder(model, precision):
    """Encoder を CoreML に変換し、精度を比較する"""
    print(f"=== Encoder 変換 ({precision}) ===")
    encoder = EncoderWrapper(model)
    encoder.eval()

    # ダミー入力: mel [1, T_src, 256], pos [1, T_src]
    T_src = 50
    dummy_mel = torch.randn(1, T_src, hp.n_mels)
    dummy_pos = torch.arange(1, T_src + 1).unsqueeze(0)  # [1, T_src]

    # PyTorch 出力
    with torch.no_grad():
        pt_output = encoder(dummy_mel, dummy_pos).numpy()
    print(f"PyTorch 出力 shape: {pt_output.shape}")

    # TorchScript trace
    print("torch.jit.trace 実行中...")
    traced = torch.jit.trace(encoder, (dummy_mel, dummy_pos))

    # CoreML 変換
    print("CoreML 変換中...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=ct.Shape(shape=(1, ct.RangeDim(1, 1000, default=T_src), hp.n_mels))),
            ct.TensorType(name="pos", shape=ct.Shape(shape=(1, ct.RangeDim(1, 1000, default=T_src))), dtype=np.int32),
        ],
        convert_to="mlprogram",
        compute_precision=get_compute_precision(precision),
    )

    if precision == "int8":
        print("Int8 量子化中...")
        mlmodel = quantize_int8(mlmodel)

    mlmodel.save(ENCODER_OUTPUT_PATH)
    print(f"保存完了: {ENCODER_OUTPUT_PATH}")

    # 精度比較
    print("出力比較中...")
    prediction = mlmodel.predict({"mel": dummy_mel.numpy(), "pos": dummy_pos.to(torch.int32).numpy()})
    output_key = list(prediction.keys())[0]
    coreml_output = prediction[output_key]

    max_abs_error = np.max(np.abs(pt_output - coreml_output))
    is_close = np.allclose(pt_output, coreml_output, atol=1e-4)
    print(f"最大絶対誤差: {max_abs_error:.6e}")
    print(f"np.allclose(atol=1e-4): {is_close}")
    print()

    return mlmodel


def convert_decoder(model, precision):
    """Decoder を CoreML に変換し、精度を比較する"""
    print(f"=== Decoder 変換 ({precision}) ===")
    decoder = DecoderWrapper(model)
    decoder.eval()

    # ダミー入力
    T_src = 50
    T_trg = 30
    dummy_memory = torch.randn(1, T_src, hp.hidden_size)  # Encoder 出力
    dummy_dec_input = torch.randn(1, T_trg, hp.n_mels)
    dummy_pos = torch.arange(1, T_trg + 1).unsqueeze(0)

    # PyTorch 出力
    with torch.no_grad():
        mel_out, postnet_out = decoder(dummy_memory, dummy_dec_input, dummy_pos)
        pt_mel = mel_out.numpy()
        pt_postnet = postnet_out.numpy()
    print(f"PyTorch mel 出力 shape: {pt_mel.shape}")
    print(f"PyTorch postnet 出力 shape: {pt_postnet.shape}")

    # TorchScript trace
    print("torch.jit.trace 実行中...")
    traced = torch.jit.trace(decoder, (dummy_memory, dummy_dec_input, dummy_pos))

    # CoreML 変換
    print("CoreML 変換中...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="memory", shape=ct.Shape(shape=(1, ct.RangeDim(1, 1000, default=T_src), hp.hidden_size))),
            ct.TensorType(name="decoder_input", shape=ct.Shape(shape=(1, ct.RangeDim(1, 1000, default=T_trg), hp.n_mels))),
            ct.TensorType(name="pos", shape=ct.Shape(shape=(1, ct.RangeDim(1, 1000, default=T_trg))), dtype=np.int32),
        ],
        convert_to="mlprogram",
        compute_precision=get_compute_precision(precision),
    )

    if precision == "int8":
        print("Int8 量子化中...")
        mlmodel = quantize_int8(mlmodel)

    mlmodel.save(DECODER_OUTPUT_PATH)
    print(f"保存完了: {DECODER_OUTPUT_PATH}")

    # 精度比較
    print("出力比較中...")
    prediction = mlmodel.predict({
        "memory": dummy_memory.numpy(),
        "decoder_input": dummy_dec_input.numpy(),
        "pos": dummy_pos.to(torch.int32).numpy(),
    })
    keys = sorted(prediction.keys())
    print(f"出力キー: {keys}")

    for key in keys:
        coreml_out = prediction[key]
        # mel_out と postnet_out のどちらかを特定
        ref = pt_mel if coreml_out.shape == pt_mel.shape else pt_postnet
        max_err = np.max(np.abs(ref - coreml_out))
        close = np.allclose(ref, coreml_out, atol=1e-4)
        print(f"  {key}: 最大絶対誤差={max_err:.6e}, allclose(atol=1e-4)={close}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        choices=["float16", "float32", "int8"],
        default="float16",
    )
    args = parser.parse_args()

    print(f"Transformer モデルをロード中... (精度: {args.precision})")
    model = load_model()

    convert_encoder(model, args.precision)
    convert_decoder(model, args.precision)

    print("完了")


if __name__ == "__main__":
    main()
