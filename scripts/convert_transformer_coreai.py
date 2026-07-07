# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b1",
#     "coreai-torch==0.4.0",
#     "torch==2.8.0",
#     "numpy",
#     "librosa",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Transformer (Encoder / Decoder) を Apple Core AI (.aimodel) に変換するスクリプト。

scripts/convert_transformer.py の CoreML 版を Core AI に焼き直したもの。
推論は Encoder → Decoder（自己回帰ループ）の順なので、CoreML 版と同様に
Encoder / Decoder を別々の .aimodel として変換する。

実行（uv が 3.11+ 隔離環境を自動構築。PronounSE の 3.9 venv では coreai-torch 不可）:

    uv run scripts/convert_transformer_coreai.py --dtype float16 --frames 262

注意:
- `.aimodel` の推論には macOS/iOS 27 が必要。Mac (macOS 26) では **エクスポートのみ**。
  数値照合（PSNR）は実機 iPhone@27 で別途。ここでは参照入出力を npy 保存する。
- Neural Engine は静的 shape 要求のため frames（T）は固定（既定 262）。
- pos は int32（NE は si32 を想定）。
"""

import argparse
import contextlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "PronounSE" / "Transformer"))
from network import Model  # noqa: E402
import hyperparams as hp  # noqa: E402

TRANSFORMER_CHKPT = (
    PROJECT_ROOT / "PronounSE" / "Transformer" / "chkpt" / "chkpt__20000.pth.tar"
)


class EncoderWrapper(torch.nn.Module):
    """eval 時は c_mask=None・attns 不要なので memory のみ返す。"""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, mel, pos):
        memory, _, _ = self.encoder(mel, pos)
        return memory


class DecoderWrapper(torch.nn.Module):
    """eval 時は c_mask=None。mel_out / postnet_out を返す。"""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, memory, decoder_input, pos):
        mel_out, postnet_out, _, _ = self.decoder(memory, decoder_input, None, pos)
        return mel_out, postnet_out


def load_model(dtype: torch.dtype):
    model = Model(hp.prenet_type).cpu()
    chkpt = torch.load(TRANSFORMER_CHKPT, map_location="cpu")
    model.load_state_dict(chkpt["model"])
    model.eval()
    if dtype == torch.float16:
        model.half()
    return model


def build_metadata(part: str) -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "Jinmaro / PronounSE"
    metadata.license = "see PronounSE submodule"
    metadata.model_description = (
        f"PronounSE Transformer {part}. Exported for on-device ANE breakdown study."
    )
    metadata.creation_date = int(time.time())
    return metadata


def save_reference(ref_dir: Path, name: str, arrays: dict) -> None:
    ref_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        np.save(ref_dir / f"{name}_{key}.npy", arr.float().cpu().numpy()
                if arr.is_floating_point() else arr.cpu().numpy())
    print(f"[INFO] 参照 {name} を保存: {ref_dir}")


def export_and_save(wrapper, args_tuple, dynamic_shapes, input_names, output_names,
                    dtype, out_path: Path, overwrite: bool, optimize: bool, part: str):
    print(f"[INFO] {part}: torch.export 実行中（静的 shape）...")
    # autocast は fp16 のときのみ（CPU autocast は bf16/fp16 のみ対応）
    ctx = torch.autocast(device_type="cpu", dtype=dtype) if dtype == torch.float16 \
        else contextlib.nullcontext()
    with ctx:
        exported = torch.export.export(
            wrapper, args=args_tuple, dynamic_shapes=dynamic_shapes
        )
    exported = exported.run_decompositions(get_decomp_table())

    print(f"[INFO] {part}: Core AI へ変換中...")
    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=input_names,
        output_names=output_names,
    )
    coreai_program = converter.to_coreai()
    if optimize:
        coreai_program.optimize()

    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} 既存。--overwrite で上書き。")
        shutil.rmtree(out_path) if out_path.is_dir() else out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coreai_program.save_asset(out_path, build_metadata(part))
    print(f"[INFO] {part}: 保存完了: {out_path}")


def convert(output_dir: Path, dtype: torch.dtype, frames: int,
            overwrite: bool, optimize: bool, save_ref: bool):
    dtype_name = str(dtype).split(".")[-1]
    ref_dir = output_dir / "reference"
    print(f"\n=== Transformer 変換: dtype={dtype_name}, frames={frames} ===")
    model = load_model(dtype)

    # ---- Encoder ----
    mel = torch.randn(1, frames, hp.n_mels, dtype=dtype)
    pos = torch.arange(1, frames + 1, dtype=torch.int32).unsqueeze(0)
    enc = EncoderWrapper(model).eval()
    with torch.no_grad():
        memory = enc(mel, pos)
    print(f"[INFO] Encoder memory shape: {tuple(memory.shape)}")
    if save_ref:
        save_reference(ref_dir, f"enc_{dtype_name}_f{frames}",
                       {"mel": mel, "pos": pos, "memory": memory})
    export_and_save(
        enc, (mel, pos),
        dynamic_shapes={"mel": {}, "pos": {}},
        input_names=["mel", "pos"], output_names=["memory"],
        dtype=dtype,
        out_path=output_dir / f"Transformer_Encoder_{dtype_name}_f{frames}.aimodel",
        overwrite=overwrite, optimize=optimize, part="Encoder",
    )

    # ---- Decoder ----
    dec_input = torch.randn(1, frames, hp.n_mels, dtype=dtype)
    dec_pos = torch.arange(1, frames + 1, dtype=torch.int32).unsqueeze(0)
    dec = DecoderWrapper(model).eval()
    with torch.no_grad():
        mel_out, postnet_out = dec(memory, dec_input, dec_pos)
    print(f"[INFO] Decoder mel_out={tuple(mel_out.shape)}, "
          f"postnet_out={tuple(postnet_out.shape)}")
    if save_ref:
        save_reference(ref_dir, f"dec_{dtype_name}_f{frames}",
                       {"memory": memory, "decoder_input": dec_input, "pos": dec_pos,
                        "mel_out": mel_out, "postnet_out": postnet_out})
    export_and_save(
        dec, (memory, dec_input, dec_pos),
        dynamic_shapes={"memory": {}, "decoder_input": {}, "pos": {}},
        input_names=["memory", "decoder_input", "pos"],
        output_names=["mel_out", "postnet_out"],
        dtype=dtype,
        out_path=output_dir / f"Transformer_Decoder_{dtype_name}_f{frames}.aimodel",
        overwrite=overwrite, optimize=optimize, part="Decoder",
    )
    print("[INFO] 数値照合（PSNR）は macOS 27 / 実機 iPhone@27 で別途実施する。")


def main():
    parser = argparse.ArgumentParser(
        description="PronounSE Transformer を Core AI (.aimodel) に変換する。"
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--frames", type=int, default=262)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "coreai_exports"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--no-reference", action="store_true")
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    convert(
        output_dir=Path(args.output_dir),
        dtype=dtype,
        frames=args.frames,
        overwrite=args.overwrite,
        optimize=not args.no_optimize,
        save_ref=not args.no_reference,
    )


if __name__ == "__main__":
    main()
