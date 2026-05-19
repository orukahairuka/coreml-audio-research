#!/usr/bin/env python3
"""debug snapshot の `waveform_postdeemph.npy` を WAV (16-bit PCM) に変換する。

Unit Test (Swift Testing) は AudioPlayer を通らないので `output_*.wav` が更新されない。
このスクリプトは debug snapshot 配下の npy を wav に変換して、再生可能な状態にする。

クリッピング処理: float 値が ±1.0 を超える場合は clamp する。
クリップした場合は標準出力に警告を出す。
PCM 16-bit, 22050 Hz, mono.

使い方:
    scripts/npy_to_wav.py audio/iPhone_3_phase1_20260519/debug/<runId>/  # 単一 run
    scripts/npy_to_wav.py audio/iPhone_3_phase1_20260519/debug/         # 配下全 run
    scripts/npy_to_wav.py --label '*Repeat1*' audio/<archive>/debug/    # ラベル filter

出力: 各 run dir に `playable.wav` を作る（既存があれば上書き）。
ラベルマッチで複数 run があるときは、ファイル名末尾に <runId> を付ける。
"""

from __future__ import annotations

import argparse
import fnmatch
import struct
import sys
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 22050


def npy_to_wav(npy_path: Path, wav_path: Path) -> dict:
    arr = np.load(npy_path).astype(np.float32).flatten()
    n = arr.size

    clipped_high = int((arr > 1.0).sum())
    clipped_low = int((arr < -1.0).sum())
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    if nan_count or inf_count:
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    clamped = np.clip(arr, -1.0, 1.0)
    int16 = (clamped * 32767).astype(np.int16)

    with wave.open(str(wav_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(int16.tobytes())

    peak = int(np.max(np.abs(int16)))
    rms = float(np.sqrt(np.mean(int16.astype(np.float64) ** 2)))

    return {
        "samples": n,
        "duration_s": n / SAMPLE_RATE,
        "clipped_high": clipped_high,
        "clipped_low": clipped_low,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "int16_rms": rms,
        "int16_peak": peak,
    }


def find_runs(root: Path, label_glob: str | None) -> list[Path]:
    """root が単一 run dir なら [root]、それ以外なら配下の run dirs を返す。"""
    if (root / "waveform_postdeemph.npy").exists():
        return [root]
    runs = []
    for entry in sorted(root.iterdir()):
        if (entry / "waveform_postdeemph.npy").exists():
            if label_glob is None or fnmatch.fnmatch(entry.name, label_glob):
                runs.append(entry)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--label", default=None, help="run dir 名の glob filter (例: '*directRun1*')")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="まとめて 1 つのフォルダに wav を集める場合の出力先")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"# not found: {args.root}", file=sys.stderr)
        return 1

    runs = find_runs(args.root, args.label)
    if not runs:
        print(f"# no waveform_postdeemph.npy found under {args.root}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# converting {len(runs)} run(s)")
    print(f"# {'run':50s} {'samples':>7s} {'rms':>8s} {'peak':>8s} {'clip+':>6s} {'clip-':>6s} {'nan':>4s}")
    for run_dir in runs:
        npy = run_dir / "waveform_postdeemph.npy"
        if out_dir:
            wav_path = out_dir / f"{run_dir.name}.wav"
        else:
            wav_path = run_dir / "playable.wav"
        stats = npy_to_wav(npy, wav_path)
        print(f"  {run_dir.name:50s} {stats['samples']:>7d} {stats['int16_rms']:>8.1f} "
              f"{stats['int16_peak']:>8d} {stats['clipped_high']:>6d} {stats['clipped_low']:>6d} "
              f"{stats['nan_count']:>4d}")
    if out_dir:
        print(f"# wrote {len(runs)} wavs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
