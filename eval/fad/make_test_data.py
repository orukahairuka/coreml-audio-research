#!/usr/bin/env python3
"""compute_fad.py の動作確認用の合成音を作る（実際の評価には使わない）。

効果音に似た「減衰する帯域雑音＋減衰する正弦波」を乱数で作り、次の集合を書き出す。
  ref/        参照（44.1 kHz、ステレオ。--common-sr の変換とモノラル化を通す）
  ref_copy/   ref と同じ音を 22,050 Hz・モノラルにしたもの（FAD は 0 に近いはず）
  same/       ref と同じ作り方で乱数だけ違う音（22,050 Hz）
  noisy/      same に白色雑音を足したもの（same より FAD が大きいはず）
  clipped/    same を 4 倍にして ±1 で切ったもの（same より FAD が大きいはず）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
import soundfile as sf


def make_clip(rng: np.random.Generator, sr: int, dur: float) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n) / sr
    lo = rng.uniform(100, 2000)
    hi = min(lo * rng.uniform(1.5, 6), sr / 2 * 0.9)
    b, a = scipy.signal.butter(4, [lo, hi], btype="band", fs=sr)
    noise = scipy.signal.lfilter(b, a, rng.standard_normal(n))
    noise /= np.max(np.abs(noise)) + 1e-9
    tone = np.sin(2 * np.pi * rng.uniform(80, 1500) * t)
    env = np.exp(-t * rng.uniform(1.5, 8))
    onset = int(rng.uniform(0, 0.3) * sr)
    x = (rng.uniform(0.3, 1.0) * noise + rng.uniform(0, 0.6) * tone) * env
    x = np.concatenate([np.zeros(onset), x])[:n]
    return (0.5 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--dur", type=float, default=3.04)
    args = parser.parse_args()

    dirs = {k: args.out / k for k in ["ref", "ref_copy", "same", "noisy", "clipped"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    rng_ref, rng_eval, rng_noise = (np.random.default_rng(s) for s in (1, 2, 3))
    for i in range(args.n):
        r = make_clip(rng_ref, 44100, args.dur)
        sf.write(dirs["ref"] / f"{i:03d}.wav", np.stack([r, r], axis=1), 44100, subtype="PCM_16")
        r22 = librosa.resample(r, orig_sr=44100, target_sr=22050, res_type="soxr_hq")
        sf.write(dirs["ref_copy"] / f"{i:03d}.wav", r22, 22050, subtype="PCM_16")
        s = make_clip(rng_eval, 22050, args.dur)
        sf.write(dirs["same"] / f"{i:03d}.wav", s, 22050, subtype="PCM_16")
        sf.write(dirs["noisy"] / f"{i:03d}.wav", s + 0.05 * rng_noise.standard_normal(s.shape).astype(np.float32),
                 22050, subtype="PCM_16")
        sf.write(dirs["clipped"] / f"{i:03d}.wav", np.clip(4 * s, -1, 1), 22050, subtype="PCM_16")
    print(f"→ {args.out}（各 {args.n} 個）")


if __name__ == "__main__":
    main()
