#!/usr/bin/env python3
"""一括合成（run_formal_eval.py）の出力を、同じ入力の PyTorch FP32 と比べる

PyTorch 側は正式評価と同じ条件（librosa.load(sr=22050)、先頭の無音を削らない・フェードしない、Decoder を T 回、
HiFi-GAN、lfilter によるデエンファシス）で合成し、--cache に保存して使い回す。
比べるのは iOS の float32 の npy（PCM16 にする前。クリッピングで差が隠れないようにする）。

出力: 各セッションのフォルダに pytorch_comparison.csv、標準出力に要約

使い方
  PronounSE/venv/bin/python scripts/compare_formal_to_pytorch.py --source ~/Desktop/Vocal_Mimicry \
      --session metrics/formal_eval/<session>/device [--session ...]
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from convert_hifigan import load_generator  # noqa: E402
from convert_transformer import load_model  # noqa: E402
from identify_original_preprocess import spectrogram  # noqa: E402
from verify_variable_length import deemphasis, run_pytorch  # noqa: E402


def logmel(x: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=22050, n_fft=1024, hop_length=256, n_mels=80)
    return 10 * np.log10(np.maximum(m, 1e-10))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="Vocal_Mimicry（元の口真似音）")
    ap.add_argument("--session", type=Path, action="append", required=True, help="取り出したセッションの device フォルダ")
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "metrics/pytorch_ref_fp32_fullinput")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    model = generator = None

    for session in args.session:
        rows = [r for r in csv.DictReader(open(session / "results.csv")) if r["status"] == "ok"]
        out = []
        for r in rows:
            stem = f"{r['reference']}_{r['speaker']}"
            ref_path = args.cache / r["speaker"] / f"{stem}_wave.npy"
            if not ref_path.exists():
                if model is None:
                    model, generator = load_model(), load_generator()
                mel = spectrogram(str(args.source / r["speaker"] / f"{stem}.wav"), trim=False, fade=False)
                _, raw = run_pytorch(model, generator, np.ascontiguousarray(mel, dtype=np.float32))
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(ref_path, deemphasis(raw))
            pt = np.load(ref_path).ravel()
            ios = np.load(session / r["output_f32"]).ravel()
            e = {"input": stem, "T": int(r["T"]), "same_length": pt.size == ios.size}
            if pt.size == ios.size:
                d = pt.astype(np.float64) - ios
                e["snr_db"] = float(10 * np.log10(np.sum(pt.astype(np.float64) ** 2) / max(np.sum(d ** 2), 1e-300)))
                e["logmel_mae_db"] = float(np.abs(logmel(pt) - logmel(ios)).mean())
            e["ios_peak"] = float(np.abs(ios).max())
            e["pytorch_peak"] = float(np.abs(pt).max())
            e["ios_clipped"] = int((np.abs(ios) >= 1).sum())
            e["pytorch_clipped"] = int((np.abs(pt) >= 1).sum())
            out.append(e)
        with open(session / "pytorch_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        ok = [e for e in out if e["same_length"]]
        lm = [e["logmel_mae_db"] for e in ok]
        snr = [e["snr_db"] for e in ok]
        crash = [e for e in ok if 65 <= e["T"] <= 80]
        print(f"{session.parent.name}: n={len(out)} 長さ一致 {len(ok)}/{len(out)} "
              f"ログメル差 中央 {st.median(lm):.2f} / 最大 {max(lm):.2f} dB  SNR 中央 {st.median(snr):.1f} / 最小 {min(snr):.1f} dB"
              + (f"  T=65〜80（{len(crash)} 件）ログメル差 最大 {max(e['logmel_mae_db'] for e in crash):.2f} dB、"
                 f"SNR 最小 {min(e['snr_db'] for e in crash):.1f} dB" if crash else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
