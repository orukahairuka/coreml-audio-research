#!/usr/bin/env python3
"""iOS の前処理（AudioFeatureExtractor）を Mac でビルドして動かし、オリジナル（PronounSE の get_spectrograms）と比べる

比べる組み合わせ
  - fullInput（iOS）    ↔ オリジナルの合成音を作った条件（get_spectrograms から preprocess() を除いたもの）
  - trimAndFade（iOS）  ↔ 公開実装の get_spectrograms そのもの（従来の比較。回帰の確認用）
入力の経路
  - 48k  : データセットの wav（48 kHz）をそのまま渡す。iOS は AVAudioConverter で 22050 Hz に変換する
  - 22k  : librosa.load(sr=22050) の結果を float32 の wav にしたもの（作業用のコピー。--out の下に作る）を渡す。
           iOS は変換しないので、読み込み後の波形は librosa と同じになるはず
出力のサンプル数（T × 256）と、提供されたオリジナルの合成音のサンプル数も比べる。

使い方
  PronounSE/venv/bin/python scripts/verify_fullinput_preprocess.py \
      --input s0p5=<口真似音 wav> ... --dataset ~/Desktop --out metrics/preprocess_fullinput_<日付>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "PronounSE/Transformer"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from utils import get_spectrograms  # noqa: E402

from identify_original_preprocess import spectrogram  # noqa: E402

AUDIO_DIR = REPO_ROOT / "ios/CoreMLAudioApp/CoreMLAudioApp/Models/Audio"
SOURCES = ["AudioFeatureExtractor", "SoxrHQDecimator", "SoxrHQDecimatorCoefficients", "NpyWriter", "SynthesisOptimizations"]


def build_and_run(out: Path, specs: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "preprocess_fullinput"
        sources = [str(REPO_ROOT / "scripts/preprocess_fullinput/main.swift")] + [str(AUDIO_DIR / f"{n}.swift") for n in SOURCES]
        subprocess.run(["swiftc", "-O", "-swift-version", "5", "-o", str(binary), *sources],
                       check=True, stdin=subprocess.DEVNULL)
        subprocess.run([str(binary), str(out), *specs], check=True, stdin=subprocess.DEVNULL)


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"same_shape": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"same_shape": True, "max_abs": float(d.max()), "mae": float(d.mean()), "identical": bool(d.max() == 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True, metavar="NAME=WAV")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    swift_out = args.out / "swift"
    wav22 = args.out / "input_22050"
    wav22.mkdir(parents=True, exist_ok=True)

    inputs = [s.partition("=")[::2] for s in args.input]
    specs = []
    for name, path in inputs:
        y, _ = librosa.load(path, sr=22050)
        p22 = wav22 / f"{name}.wav"
        sf.write(p22, y, 22050, subtype="FLOAT")
        specs += [f"{name}_48k={path}", f"{name}_22k={p22}"]
    build_and_run(swift_out, specs)

    report = []
    for name, path in inputs:
        y, _ = librosa.load(path, sr=22050)
        ref_full = spectrogram(path, trim=False, fade=False)
        ref_public = get_spectrograms(path)[0]
        spk = os.path.basename(os.path.dirname(path))
        orig = os.path.join(args.dataset, "PronounSE", spk, os.path.basename(path).replace(".wav", "_pronounse.wav"))
        orig_samples = sf.info(orig).frames
        entry = {"name": name, "wav": path, "input_samples_22050": int(len(y)),
                 "expected_T_full": 1 + len(y) // 256, "original_samples": int(orig_samples), "routes": {}}
        print(f"[{name}] N={len(y)}  期待 T=1+floor(N/256)={entry['expected_T_full']}  オリジナル {orig_samples} サンプル")
        for route in ["48k", "22k"]:
            load = np.load(swift_out / f"{name}_{route}_load.npy").ravel()
            full = np.load(swift_out / f"{name}_{route}_fullInput_mel.npy")
            public = np.load(swift_out / f"{name}_{route}_trimAndFade_mel.npy")
            r = {"load_samples": int(load.size), "load_vs_librosa": diff(y, load),
                 "fullInput_T": int(full.shape[0]), "fullInput_vs_original_condition": diff(ref_full, full),
                 "trimAndFade_T": int(public.shape[0]), "trimAndFade_vs_get_spectrograms": diff(ref_public, public),
                 "output_samples_fullInput": int(full.shape[0]) * 256}
            r["output_samples_match_original"] = r["output_samples_fullInput"] == orig_samples
            entry["routes"][route] = r
            f, l = r["fullInput_vs_original_condition"], r["load_vs_librosa"]
            fmt = (lambda d: f"max|差| {d['max_abs']:.2e} 平均 {d['mae']:.2e}" if d["same_shape"]
                   else f"形が違う {d['shape_a']} vs {d['shape_b']}")
            print(f"  {route}: 読み込み {r['load_samples']} サンプル（{fmt(l)}）  fullInput T={r['fullInput_T']} "
                  f"メル {fmt(f)}  出力 {r['output_samples_fullInput']} サンプル（オリジナルと一致 {r['output_samples_match_original']}）"
                  f"  | trimAndFade T={r['trimAndFade_T']} メル {fmt(r['trimAndFade_vs_get_spectrograms'])}")
        report.append(entry)
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
