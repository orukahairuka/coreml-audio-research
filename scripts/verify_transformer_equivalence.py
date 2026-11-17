#!/usr/bin/env python3
"""PyTorch 版と Core ML 版（Float32・CPU）の Transformer Encoder / Decoder の同等性を検証する。

モデル移植の妥当性の確認が目的（音質の評価ではない）。両者に「完全に同じ正規化済みメル」を入れ、
固定長化（fixed262 の 0 埋め）による差と、Core ML 変換による差を分けて測る。

比べる 3 種類:
  pt_var : PyTorch・可変長（T' フレームのまま。PronounSE の synthesis.py と同じ）
  pt_fix : PyTorch・262 フレームに 0 埋め（pos は 1..262。iOS の EncoderRunner / DecoderRunner と同じ）
  ml_fix : Core ML（*_float32_fixed262.mlpackage, CPU_ONLY）。入力は pt_fix と同じ

1. Encoder 出力（先頭 T' 行）: pt_var vs pt_fix（固定長化）、pt_fix vs ml_fix（変換）
2. Decoder 1 ステップ比較（全ステップ）: 全ステップで pt_var の自己回帰で生成した同じ系列を入れ、
   そのステップで使う出力フレーム（mel_out の行 s）を比べる
     pt_var        : memory = pt_var の Encoder 出力、decoder_input は長さ s+1
     pt_var_decpad : memory は同じ、decoder_input だけ 262 に 0 埋め（Decoder 入力の 0 埋めの影響）
     pt_fix        : memory = pt_fix の Encoder 出力（262 行）、decoder_input も 262 に 0 埋め
     ml_fix        : pt_fix と同じ memory・decoder_input を Core ML に入れる（変換の影響）
3. 自己回帰（それぞれ自分の出力を次の入力にして最後まで回す）: ml_fix は Core ML の Encoder から通す。
   各ステップの出力フレームと、最終 Postnet 出力（先頭 T' 行。末尾 10 フレームは別に集計）を比べる

入力は iOS の計測モードが保存した Swift の前処理のメル（`*_mel_in.npy`, [T', 256]）。
Swift と PyTorch の前処理の違いはここでは扱わない（両者に同じメルを入れる）。

実行例（PronounSE の venv）:
    PronounSE/venv/bin/python scripts/verify_transformer_equivalence.py \\
        --mel-dir metrics/thesis_ch4v4_decinput_iPhone_3_20261118_001315/bulk/raw/device/runs/L0001_infer \\
        --out-dir metrics/transformer_equivalence_<日時>
"""

from __future__ import annotations

import warnings
warnings.simplefilter("ignore")

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch as t

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# network.py が `from module import *` するので Transformer ディレクトリも通す（extract_pytorch_stages.py と同じ）
sys.path.insert(0, str(REPO_ROOT / "PronounSE"))
sys.path.insert(0, str(REPO_ROOT / "PronounSE" / "Transformer"))
from generate_decoder_reference import load_model  # noqa: E402
from Transformer import hyperparams as hp  # noqa: E402

import coremltools as ct  # noqa: E402

FIXED_T = 262
MODEL_DIR = REPO_ROOT / "ios/CoreMLAudioApp/CoreMLAudioApp/MLModels"
POSTNET_TAIL = 10  # Postnet の畳み込みが 0 埋め側を見うる末尾のフレーム数（集計を分けるだけ）


# ---------------------------------------------------------------- 指標

def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.abs(a - b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
    return {"mae": float(d.mean()), "max_abs": float(d.max()), "cosine": cos}


def per_step(rows_a: np.ndarray, rows_b: np.ndarray) -> dict:
    """[S, 256] 同士を行（ステップ）ごとに比べ、ステップ全体で集計する"""
    ms = [metrics(x, y) for x, y in zip(rows_a, rows_b)]
    mae = np.array([m["mae"] for m in ms])
    mx = np.array([m["max_abs"] for m in ms])
    cos = np.array([m["cosine"] for m in ms])
    return {
        "steps": len(ms),
        "mae_mean": float(mae.mean()), "mae_max": float(mae.max()),
        "max_abs": float(mx.max()), "max_abs_step": int(mx.argmax()),
        "cosine_min": float(np.nanmin(cos)), "cosine_mean": float(np.nanmean(cos)),
    }


# ---------------------------------------------------------------- モデルの呼び出し

class Runner:
    def __init__(self, checkpoint: Path):
        t.set_num_threads(1)  # CPU の並列化による演算順の揺れを避ける
        self.model = load_model(checkpoint, "cpu")
        cu = ct.ComputeUnit.CPU_ONLY
        self.ml_enc = ct.models.MLModel(str(MODEL_DIR / "Transformer_Encoder_float32_fixed262.mlpackage"), compute_units=cu)
        self.ml_dec = ct.models.MLModel(str(MODEL_DIR / "Transformer_Decoder_float32_fixed262.mlpackage"), compute_units=cu)
        dec_outs = [o.name for o in self.ml_dec.get_spec().description.output]
        self.ml_postnet_key = [n for n in dec_outs if n != "mel_out"][0]
        self.ml_enc_key = self.ml_enc.get_spec().description.output[0].name

    @staticmethod
    def pos(n: int) -> np.ndarray:
        return np.arange(1, n + 1, dtype=np.int32)[None, :]

    # PyTorch（eval では c_mask=None。convert_transformer.py のラッパーと同じ呼び方）
    def pt_encoder(self, mel: np.ndarray) -> np.ndarray:
        with t.no_grad():
            out, _, _ = self.model.encoder(t.from_numpy(mel[None]), t.from_numpy(self.pos(mel.shape[0]).astype(np.int64)))
        return out[0].numpy()

    def pt_decoder(self, memory: np.ndarray, dec_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with t.no_grad():
            mel_out, postnet_out, _, _ = self.model.decoder(
                t.from_numpy(memory[None]), t.from_numpy(dec_in[None]), None,
                t.from_numpy(self.pos(dec_in.shape[0]).astype(np.int64)))
        return mel_out[0].numpy(), postnet_out[0].numpy()

    # Core ML（固定 262）
    def ml_encoder(self, mel262: np.ndarray) -> np.ndarray:
        out = self.ml_enc.predict({"mel": mel262[None].astype(np.float32), "pos": self.pos(FIXED_T)})
        return np.asarray(out[self.ml_enc_key], dtype=np.float32)[0]

    def ml_decoder(self, memory262: np.ndarray, dec_in262: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self.ml_dec.predict({
            "memory": memory262[None].astype(np.float32),
            "decoder_input": dec_in262[None].astype(np.float32),
            "pos": self.pos(FIXED_T),
        })
        return (np.asarray(out["mel_out"], dtype=np.float32)[0],
                np.asarray(out[self.ml_postnet_key], dtype=np.float32)[0])


def pad262(x: np.ndarray) -> np.ndarray:
    out = np.zeros((FIXED_T, x.shape[1]), dtype=np.float32)
    out[: x.shape[0]] = x
    return out


# ---------------------------------------------------------------- 検証本体

def verify_input(r: Runner, mel: np.ndarray) -> dict:
    T = mel.shape[0]
    assert T <= FIXED_T and mel.shape[1] == hp.n_mels
    res: dict = {"T_prime": T}

    # 1. Encoder
    enc_var = r.pt_encoder(mel)                 # [T, 512]
    enc_fix_pt = r.pt_encoder(pad262(mel))      # [262, 512]
    enc_fix_ml = r.ml_encoder(pad262(mel))      # [262, 512]
    res["encoder"] = {
        "pt_var_vs_pt_fix": metrics(enc_var, enc_fix_pt[:T]),
        "pt_fix_vs_ml_fix": metrics(enc_fix_pt[:T], enc_fix_ml[:T]),
        "pt_var_vs_ml_fix": metrics(enc_var, enc_fix_ml[:T]),
        # 0 埋め側の行も含めた 262 行全体（Decoder の dot-attention はこの全行を見る）
        "pt_fix_vs_ml_fix_all262": metrics(enc_fix_pt, enc_fix_ml),
    }

    # 3a. pt_var の自己回帰（1 ステップ比較の「同じ生成済み系列」にも使う）
    t0 = time.time()
    seq = np.zeros((1, hp.n_mels), dtype=np.float32)   # 先頭はゼロフレーム
    rows_var, post_var = [], None
    for s in range(T):
        mel_out, post = r.pt_decoder(enc_var, seq)
        rows_var.append(mel_out[s])
        seq = np.concatenate([seq, mel_out[s:s + 1]], axis=0)
        post_var = post
    rows_var = np.stack(rows_var)
    teacher = seq[:T]  # ステップ s の入力は teacher[: s + 1]

    # 2. 1 ステップ比較（全ステップで同じ生成済み系列を入れる）
    one = {"pt_var": [], "pt_var_decpad": [], "pt_fix": [], "ml_fix": []}
    for s in range(T):
        dec_in = teacher[: s + 1]
        one["pt_var"].append(r.pt_decoder(enc_var, dec_in)[0][s])
        one["pt_var_decpad"].append(r.pt_decoder(enc_var, pad262(dec_in))[0][s])
        one["pt_fix"].append(r.pt_decoder(enc_fix_pt, pad262(dec_in))[0][s])
        one["ml_fix"].append(r.ml_decoder(enc_fix_pt, pad262(dec_in))[0][s])
    one = {k: np.stack(v) for k, v in one.items()}
    res["decoder_one_step"] = {
        "decoder_input_padding (pt_var vs pt_var_decpad)": per_step(one["pt_var"], one["pt_var_decpad"]),
        "encoder_memory_padding (pt_var_decpad vs pt_fix)": per_step(one["pt_var_decpad"], one["pt_fix"]),
        "conversion (pt_fix vs ml_fix)": per_step(one["pt_fix"], one["ml_fix"]),
        "total (pt_var vs ml_fix)": per_step(one["pt_var"], one["ml_fix"]),
    }

    # 3b. pt_fix / ml_fix の自己回帰
    def run_fixed(decoder, memory262):
        seq = np.zeros((1, hp.n_mels), dtype=np.float32)
        rows, post = [], None
        for s in range(T):
            mel_out, post = decoder(memory262, pad262(seq))
            rows.append(mel_out[s])
            seq = np.concatenate([seq, mel_out[s:s + 1]], axis=0)
        return np.stack(rows), post[:T]

    rows_fix_pt, post_fix_pt = run_fixed(r.pt_decoder, enc_fix_pt)
    rows_fix_ml, post_fix_ml = run_fixed(r.ml_decoder, enc_fix_ml)  # Core ML は Encoder から通す
    head = max(T - POSTNET_TAIL, 0)

    def ar_compare(a_rows, b_rows, a_post, b_post):
        q = {f"step_{p}": metrics(a_rows[p], b_rows[p]) for p in sorted({0, T // 4, T // 2, 3 * T // 4, T - 1})}
        return {
            "frames_all_steps": per_step(a_rows, b_rows),
            "frames_at": q,
            "postnet_all": metrics(a_post, b_post),
            "postnet_head": metrics(a_post[:head], b_post[:head]) if head > 0 else None,
            "postnet_tail10": metrics(a_post[head:], b_post[head:]),
        }

    res["autoregressive"] = {
        "fixed_length (pt_var vs pt_fix)": ar_compare(rows_var, rows_fix_pt, post_var, post_fix_pt),
        "conversion (pt_fix vs ml_fix)": ar_compare(rows_fix_pt, rows_fix_ml, post_fix_pt, post_fix_ml),
        "total (pt_var vs ml_fix)": ar_compare(rows_var, rows_fix_ml, post_var, post_fix_ml),
    }
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res, {"post_var": post_var, "post_fix_pt": post_fix_pt, "post_fix_ml": post_fix_ml,
                 "rows_var": rows_var, "rows_fix_pt": rows_fix_pt, "rows_fix_ml": rows_fix_ml,
                 "enc_var": enc_var, "enc_fix_pt": enc_fix_pt, "enc_fix_ml": enc_fix_ml}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mel-dir", required=True, type=Path, help="*_in<名前>_r0_mel_in.npy がある runs/<launch> ディレクトリ")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--inputs", default="1,2,6,11")
    ap.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "PronounSE/Transformer/chkpt/chkpt__20000.pth.tar")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    r = Runner(args.checkpoint)
    summary = {"torch": t.__version__, "coremltools": ct.__version__, "fixed_T": FIXED_T,
               "mel_dir": str(args.mel_dir), "inputs": {}}
    for name in args.inputs.split(","):
        paths = [p for p in args.mel_dir.glob("*_mel_in.npy") if re.search(rf"_in{name}_r0_mel_in\.npy$", p.name)]
        assert len(paths) == 1, (name, paths)
        mel = np.load(paths[0]).astype(np.float32)
        sha = hashlib.sha256(mel.astype("<f4").tobytes()).hexdigest()
        print(f"# input {name}: {paths[0].name} shape={mel.shape} sha256={sha[:12]}", flush=True)
        res, arrays = verify_input(r, mel)
        res["mel_sha256"] = sha
        res["mel_file"] = paths[0].name
        summary["inputs"][name] = res
        np.savez_compressed(args.out_dir / f"in{name}_arrays.npz", mel=mel, **arrays)
        print(f"  done in {res['elapsed_sec']}s", flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"# wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
