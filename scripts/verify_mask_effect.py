#!/usr/bin/env python3
"""固定長化の差に対する、0 埋め部分のマスクの効果を PyTorch 版で確かめる（卒論 4-2 節）。

T' が 262 に満たない入力について、262 フレームに 0 埋めしたモデルの出力を、可変長の出力と比べる。

  (a) マスクなし（アプリと同じ構成）
  (b) Encoder の Self-Attention のみにマスク
  (c) Encoder の Self-Attention と Decoder の Cross-Attention の両方にマスク

それぞれ自身の出力を次の入力として T' 回繰り返し、最後の回の補正後の出力の先頭 T' フレームを比べる。
入力は iOS の計測モードが保存した Swift の前処理のメル（`*_r0_mel_in.npy`）。

実行例（PronounSE の venv）:
    PronounSE/venv/bin/python scripts/verify_mask_effect.py \\
        --mel-dir metrics/final_ch4v7_<日時>/iPhone_13/e2e/noprofile/raw/device/runs/L0001_infer \\
        --out-dir metrics/final_ch4v7_<日時>/mask_effect/new_mel
"""

from __future__ import annotations

import warnings
warnings.simplefilter("ignore")

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch as t

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# network.py が `from module import *` するので Transformer ディレクトリも通す
sys.path.insert(0, str(REPO_ROOT / "PronounSE"))
sys.path.insert(0, str(REPO_ROOT / "PronounSE" / "Transformer"))
from generate_decoder_reference import load_model  # noqa: E402

FIXED_T = 262
N_MELS = 256
TAIL = 10
LABELS = {"1": "入力1", "6": "入力2", "11": "入力3", "2": "入力4"}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def compare(reference: np.ndarray, output: np.ndarray) -> dict:
    return {
        "max_abs": float(np.abs(output - reference).max()),
        "cosine": cosine(output, reference),
        "tail10_max_abs": float(np.abs(output[:, -TAIL:] - reference[:, -TAIL:]).max()),
        "tail10_cosine": cosine(output[:, -TAIL:], reference[:, -TAIL:]),
    }


class MaskedModel:
    def __init__(self, checkpoint: Path):
        t.set_num_threads(1)
        model = load_model(checkpoint, "cpu")
        self.encoder = model.encoder
        self.decoder = model.decoder

    def encode(self, mel: t.Tensor, pos: t.Tensor, mask: t.Tensor | None = None) -> t.Tensor:
        encoder = self.encoder
        x = encoder.norm(encoder.encoder_prenet(mel))
        x = encoder.pos_emb(pos) * encoder.alpha + x
        for attention, ffn in zip(encoder.layers, encoder.ffns):
            x, _ = attention(x, x, mask=mask, query_mask=None)
            x = ffn(x)
        return x

    def decode(self, memory: t.Tensor, decoder_input: t.Tensor, pos: t.Tensor,
               cross_mask: t.Tensor | None) -> tuple[t.Tensor, t.Tensor]:
        decoder = self.decoder
        length = decoder_input.size(1)
        causal = t.triu(t.ones(length, length), diagonal=1).unsqueeze(0).bool()
        x = decoder.norm(decoder.decoder_prenet(decoder_input))
        x = decoder.pos_emb(pos) * decoder.alpha + x
        for self_attention, cross_attention, ffn in zip(decoder.selfattn_layers, decoder.dotattn_layers, decoder.ffns):
            x, _ = self_attention(x, x, mask=causal, query_mask=None)
            x, _ = cross_attention(memory, x, mask=cross_mask, query_mask=None)
            x = ffn(x)
        mel_out = decoder.mel_linear(x)
        postnet_input = mel_out.transpose(1, 2)
        postnet_out = (postnet_input + decoder.postconvnet(postnet_input)).transpose(1, 2)
        return mel_out, postnet_out

    def autoregress(self, memory: t.Tensor, frames: int, fixed: bool, cross_mask: t.Tensor | None = None) -> np.ndarray:
        generated = t.zeros(1, 1, N_MELS)
        postnet_out = None
        for step in range(frames):
            if fixed:
                decoder_input = t.zeros(1, FIXED_T, N_MELS)
                decoder_input[:, :generated.size(1), :] = generated
                pos = t.arange(1, FIXED_T + 1).unsqueeze(0)
                mel_out, postnet_out = self.decode(memory, decoder_input, pos, cross_mask)
            else:
                pos = t.arange(1, generated.size(1) + 1).unsqueeze(0)
                mel_out, postnet_out = self.decode(memory, generated, pos, None)
            generated = t.cat([generated, mel_out[:, step:step + 1, :]], dim=1)
        return postnet_out[:, :frames, :].numpy()


def verify(model: MaskedModel, mel: np.ndarray) -> dict:
    frames = mel.shape[0]
    assert frames < FIXED_T and mel.shape[1] == N_MELS
    with t.no_grad():
        variable = t.from_numpy(mel).unsqueeze(0)
        variable_pos = t.arange(1, frames + 1).unsqueeze(0)
        padded = t.zeros(1, FIXED_T, N_MELS)
        padded[:, :frames, :] = variable
        fixed_pos = t.arange(1, FIXED_T + 1).unsqueeze(0)
        padding_mask = t.zeros(1, FIXED_T, FIXED_T, dtype=t.bool)
        padding_mask[:, :, frames:] = True

        memory_variable = model.encode(variable, variable_pos)
        memory_fixed = model.encode(padded, fixed_pos)
        memory_masked = model.encode(padded, fixed_pos, mask=padding_mask)

        reference = model.autoregress(memory_variable, frames, fixed=False)
        outputs = {
            "a_no_mask": model.autoregress(memory_fixed, frames, fixed=True),
            "b_encoder_mask": model.autoregress(memory_masked, frames, fixed=True),
            "c_encoder_and_cross_mask": model.autoregress(memory_masked, frames, fixed=True, cross_mask=padding_mask),
        }
    encoder = {
        "no_mask": {"max_abs": float((memory_fixed[:, :frames] - memory_variable).abs().max()),
                    "cosine": cosine(memory_fixed[:, :frames].numpy(), memory_variable.numpy())},
        "encoder_mask": {"max_abs": float((memory_masked[:, :frames] - memory_variable).abs().max()),
                         "cosine": cosine(memory_masked[:, :frames].numpy(), memory_variable.numpy())},
    }
    return {"T_prime": frames, "encoder_rows": encoder,
            "postnet": {name: compare(reference, output) for name, output in outputs.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mel-dir", required=True, type=Path, help="*_in<名前>_r0_mel_in.npy がある runs/<launch> ディレクトリ")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--inputs", default="11,2", help="T' が 262 に満たない入力の名前（カンマ区切り）")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "PronounSE/Transformer/chkpt/chkpt__20000.pth.tar")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"{args.out_dir} は空ではありません（既存の結果を上書きしないため止めます）")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = MaskedModel(args.checkpoint)
    summary: dict = {"torch": t.__version__, "fixed_T": FIXED_T, "mel_dir": str(args.mel_dir), "inputs": {}}
    for name in args.inputs.split(","):
        paths = [p for p in args.mel_dir.glob("*_mel_in.npy") if re.search(rf"_in{name}_r0_mel_in\.npy$", p.name)]
        assert len(paths) == 1, (name, paths)
        mel = np.load(paths[0]).astype(np.float32)
        result = verify(model, mel)
        result["label"] = LABELS.get(name, name)
        result["mel_file"] = paths[0].name
        result["mel_sha256"] = hashlib.sha256(mel.astype("<f4").tobytes()).hexdigest()
        summary["inputs"][name] = result
        print(f"# {result['label']} T'={result['T_prime']}", flush=True)
        for condition, metrics in result["postnet"].items():
            print(f"  {condition:26s} 最大絶対誤差 {metrics['max_abs']:.3e}  コサイン類似度 {metrics['cosine']:.6f}  "
                  f"末尾10 {metrics['tail10_max_abs']:.3e} / {metrics['tail10_cosine']:.6f}", flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"# wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
