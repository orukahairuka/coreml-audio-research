# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b1",
#     "coreai-torch==0.4.0",
#     "torch==2.8.0",
#     "numpy",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""HiFi-GAN Generator を Apple Core AI (.aimodel) に変換するスクリプト。

scripts/convert_hifigan.py の CoreML 版を Core AI に焼き直したもの。
研究目的は「CoreML の ANE 破綻（HiFi-GAN×NE で rms 約4.25倍→clip）が、
別ランタイム Core AI でも再現するか」の切り分け。

実行（uv が 3.11+ 隔離環境を自動構築。PronounSE の 3.9 venv では coreai-torch が
入らないので uv 必須）:

    uv run scripts/convert_hifigan_coreai.py --dtype float16 --frames 262 \
        --mel-npy result/debug/<run>/postnet_output.npy
    uv run scripts/convert_hifigan_coreai.py --dtype float32 --frames 262 \
        --mel-npy result/debug/<run>/postnet_output.npy

    参照入力は実 mel（postnet 出力）を --mel-npy で渡す。未指定だと乱数 mel に
    フォールバックし、その場合の PSNR は「移植の数値等価性」しか測れず、本スタディの
    主目的（実 mel での HiFi-GAN×NE クリップ破綻の再現）は測れない。

注意:
- coreai-torch は Python >= 3.11 が必要（PronounSE venv の 3.9 では不可）。
- `.aimodel` の *推論* は macOS/iOS 27 が必要。Mac (macOS 26) では **エクスポートのみ**で、
  数値照合（PSNR）は実機 iPhone@27 で別途行う。ここでは後段の照合用に PyTorch 参照
  入出力を npy 保存する。
- Neural Engine は静的 shape を要求するため、フレーム数 T は固定（既定 262＝本番採用値）。
"""

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# HiFiGAN モジュールのインポートパスを通す
sys.path.insert(0, str(PROJECT_ROOT / "PronounSE" / "HiFiGAN"))
from models import Generator  # noqa: E402

HIFIGAN_CHKPT_DIR = PROJECT_ROOT / "PronounSE" / "HiFiGAN" / "chkpt"


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_generator(dtype: torch.dtype) -> torch.nn.Module:
    """Generator を CPU でロードし weight_norm を除去して返す。

    convert_hifigan.py（CoreML 版）と同じ手順。float16 指定時は params を half 化する。
    """
    with open(HIFIGAN_CHKPT_DIR / "config.json") as f:
        param = AttrDict(json.load(f))

    generator = Generator(param).cpu()
    chkpt = torch.load(HIFIGAN_CHKPT_DIR / "g_00009000", map_location="cpu")
    generator.load_state_dict(chkpt["generator"])
    generator.eval()
    generator.remove_weight_norm()
    if dtype == torch.float16:
        generator.half()
    return generator


def build_metadata() -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "Jinmaro / PronounSE"
    metadata.license = "see PronounSE submodule"
    metadata.model_description = (
        "HiFi-GAN Generator (vocoder) from PronounSE. Converts a (1, 256, T) mel "
        "spectrogram to a waveform. Exported for on-device ANE breakdown study."
    )
    metadata.creation_date = int(time.time())
    return metadata


def save_reference(input_dir: Path, dtype_name: str, frames: int,
                   mel: torch.Tensor, audio: torch.Tensor) -> None:
    """実機 PSNR 照合用に PyTorch 参照入出力を npy 保存する。"""
    ref_dir = input_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.save(ref_dir / f"hifigan_mel_{dtype_name}_f{frames}.npy",
            mel.float().cpu().numpy())
    np.save(ref_dir / f"hifigan_pt_out_{dtype_name}_f{frames}.npy",
            audio.float().cpu().numpy())
    print(f"[INFO] 参照入出力を保存: {ref_dir}")


def load_reference_mel(mel_npy: Path, frames: int, dtype: torch.dtype) -> torch.Tensor:
    """実 mel（postnet 出力）を HiFi-GAN 入力レイアウト (1, 256, T) で読む。

    実機 debug snapshot の `postnet_output.npy`（[T, 256]）や、既に
    [1, 256, T] / [1, T, 256] の npy を受け付け、いずれも (1, 256, frames) に整える。
    参照入力を乱数にすると PSNR は「移植の数値等価性」しか測れず、本スタディの主目的
    （実 mel での HiFi-GAN×NE クリップ破綻の再現）を測れないため、実 mel を渡せるようにする。
    """
    arr = np.load(mel_npy)
    t = torch.from_numpy(arr).float().squeeze()  # 余分な batch 次元を落とす
    if t.ndim != 2:
        raise ValueError(f"mel npy は 2D（[T,256] か [256,T]）を想定: {tuple(t.shape)}")
    # [256, T] へ整える（256=n_mels 側を先頭に）
    if t.shape[0] == frames and t.shape[1] == 256:
        t = t.transpose(0, 1)          # [T, 256] -> [256, T]
    elif t.shape[0] == 256 and t.shape[1] == frames:
        pass                            # 既に [256, T]
    else:
        raise ValueError(
            f"mel npy の shape {tuple(t.shape)} が frames={frames}, n_mels=256 と不整合"
        )
    return t.unsqueeze(0).to(dtype).contiguous()  # (1, 256, T)


def convert(output_dir: Path, dtype: torch.dtype, frames: int,
            overwrite: bool, optimize: bool, save_ref: bool,
            mel_npy: Path | None = None) -> None:
    dtype_name = str(dtype).split(".")[-1]
    print(f"\n=== 変換開始: dtype={dtype_name}, frames={frames} ===")

    print("[INFO] Generator をロード中...")
    generator = load_generator(dtype)

    # 静的 shape（NE 要件）。mel: (1, 256, T)
    # 参照照合を意味あるものにするため、可能なら実 mel（postnet 出力）を使う。
    # --mel-npy 未指定時のみ乱数にフォールバック（グラフ trace は shape のみ依存で成立するが、
    # その場合の PSNR は移植の等価性検証止まりで、実 mel でのクリップ破綻は測れない）。
    if mel_npy is not None:
        mel = load_reference_mel(mel_npy, frames, dtype)
        print(f"[INFO] 実 mel を参照入力に使用: {mel_npy}")
    else:
        mel = torch.randn(1, 256, frames, dtype=dtype)
        print("[WARN] --mel-npy 未指定: 乱数 mel を使用。PSNR は等価性検証のみで、"
              "実 mel でのクリップ破綻は測れない。実スタディでは --mel-npy を渡すこと。")

    with torch.no_grad():
        pt_audio = generator(mel)
    print(f"[INFO] PyTorch 出力 shape: {tuple(pt_audio.shape)}, "
          f"rms={pt_audio.float().pow(2).mean().sqrt():.4f}")

    if save_ref:
        save_reference(output_dir, dtype_name, frames, mel, pt_audio)

    print("[INFO] torch.export 実行中（静的 shape）...")
    # autocast は fp16 のときのみ（CPU autocast は bf16/fp16 のみ対応）
    ctx = torch.autocast(device_type="cpu", dtype=dtype) if dtype == torch.float16 \
        else contextlib.nullcontext()
    with ctx:
        exported = torch.export.export(
            generator, args=(mel,), dynamic_shapes={"x": {}}
        )
    exported = exported.run_decompositions(get_decomp_table())

    print("[INFO] Core AI へ変換中...")
    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["mel"],
        output_names=["audio"],
    )
    coreai_program = converter.to_coreai()
    if optimize:
        print("[INFO] optimize() 実行中...")
        coreai_program.optimize()

    model_path = output_dir / f"HiFiGAN_Generator_{dtype_name}_f{frames}.aimodel"
    if model_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{model_path} は既に存在します。--overwrite で上書きできます。"
            )
        shutil.rmtree(model_path) if model_path.is_dir() else model_path.unlink()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    coreai_program.save_asset(model_path, build_metadata())
    print(f"[INFO] 保存完了: {model_path}")
    print("[INFO] 数値照合（PSNR）は macOS 27 / 実機 iPhone@27 で別途実施する。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HiFi-GAN Generator を Core AI (.aimodel) に変換する。"
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--frames", type=int, default=262,
                        help="mel フレーム数 T（静的 shape。既定 262＝本番採用値）")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "coreai_exports"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-optimize", action="store_true",
                        help="coreai_program.optimize() を呼ばない")
    parser.add_argument("--no-reference", action="store_true",
                        help="PyTorch 参照入出力の npy 保存をしない")
    parser.add_argument("--mel-npy", default=None,
                        help="参照入力に使う実 mel の npy（例: 実機 debug snapshot の "
                             "postnet_output.npy）。未指定だと乱数 mel にフォールバックする")
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    convert(
        output_dir=Path(args.output_dir),
        dtype=dtype,
        frames=args.frames,
        overwrite=args.overwrite,
        optimize=not args.no_optimize,
        save_ref=not args.no_reference,
        mel_npy=Path(args.mel_npy) if args.mel_npy else None,
    )


if __name__ == "__main__":
    main()
