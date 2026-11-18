#!/usr/bin/env python3
"""librosa.load（soxr_hq）の 44.1 kHz → 22.05 kHz を再現する 2:1 FIR の係数を作る。

soxr_hq は線形な処理なので、単位インパルスへの応答を読み取れば FIR の係数が得られる。
出力 y[n] と入力 x[p] の関係は y[n] = Σ_p x[p] h[2n - p] で、
偶数位置と奇数位置のインパルスの応答を組み合わせると h[d] のすべての d が求まる。

サブコマンド:
  analyze   係数の再現性と、タップ数を減らしたときの誤差を調べて JSON に保存する
  generate  係数を Swift のソース（16 進の浮動小数点リテラル）に書き出す

実行例:
  PronounSE/venv/bin/python scripts/generate_resampler_fir.py analyze --out metrics/preprocess_fir/analysis.json
  PronounSE/venv/bin/python scripts/generate_resampler_fir.py generate

しきい値の既定 1e-7 の根拠: 1e-7（381 タップ）より小さくしてもメルの誤差は減らない（analyze の結果）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402
from scipy import signal  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "data/thesis_ch4/inputs"
INPUT_IDS = ["1", "6", "11", "2"]
SWIFT_PATH = REPO_ROOT / "ios/CoreMLAudioApp/CoreMLAudioApp/Models/Audio/SoxrHQDecimatorCoefficients.swift"

SOURCE_RATE = 44100
TARGET_RATE = 22050
QUALITY = "soxr_hq"


def impulse_response(length: int = 1 << 18, position: int | None = None, amplitude: float = 1.0) -> dict[int, float]:
    """h[d]（d = 2n - p）を辞書で返す"""
    p = length // 2 if position is None else position
    if p % 2:
        raise ValueError("position は偶数にする")
    h: dict[int, float] = {}
    for pos in (p, p + 1):
        x = np.zeros(length, dtype=np.float64)
        x[pos] = amplitude
        y = soxr.resample(x, SOURCE_RATE, TARGET_RATE, quality=QUALITY) / amplitude
        for n, value in enumerate(y):
            h[2 * n - pos] = float(value)
    return h


def trim(h: dict[int, float], threshold: float) -> tuple[int, np.ndarray]:
    """|h| が threshold を超える最初と最後の間を取り出す。(先頭の d, 係数) を返す"""
    ds = np.array(sorted(h))
    values = np.array([h[int(d)] for d in ds])
    keep = np.where(np.abs(values) > threshold)[0]
    lo, hi = int(ds[keep[0]]), int(ds[keep[-1]])
    return lo, values[(ds >= lo) & (ds <= hi)]


def decimate(x: np.ndarray, lo: int, coefficients: np.ndarray) -> np.ndarray:
    """y[n] = Σ_j c[j] x[2n - lo - j]。出力の長さは librosa.resample と同じ ceil(len(x) / 2)"""
    full = signal.fftconvolve(x.astype(np.float64), coefficients.astype(np.float64))
    count = int(np.ceil(len(x) / 2))
    out = np.zeros(count, dtype=np.float64)
    index = 2 * np.arange(count) - lo
    valid = (index >= 0) & (index < len(full))
    out[valid] = full[index[valid]]
    return out


def normalized_mel(y: np.ndarray) -> np.ndarray:
    """PronounSE の get_spectrograms と同じ処理（読み込みより後）"""
    sys.path.insert(0, str(REPO_ROOT / "PronounSE/Transformer"))
    from utils import preprocess  # noqa: PLC0415

    y = preprocess(y.astype(np.float32).copy())
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    magnitude = np.abs(librosa.stft(y=y, n_fft=1024, hop_length=256, win_length=1024))
    mel = librosa.filters.mel(sr=TARGET_RATE, n_fft=1024, n_mels=256) @ magnitude
    db = 20 * np.log10(np.maximum(1e-5, mel))
    return np.clip((db - 20 + 100) / 100, 1e-8, 1).T.astype(np.float32)


def versions() -> dict[str, str]:
    return {
        "soxr": soxr.__version__, "libsoxr": getattr(soxr, "__libsoxr_version__", "unknown"),
        "librosa": librosa.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "quality": QUALITY, "source_rate": str(SOURCE_RATE), "target_rate": str(TARGET_RATE),
    }


def analyze(out_path: Path) -> int:
    base = impulse_response()
    lo, reference = trim(base, 1e-9)
    report: dict = {"versions": versions(), "reproducibility": [], "truncation": []}

    # 1. 係数の再現性: インパルスの位置・信号長・振幅を変えても同じ係数になるか
    for label, kwargs in [
        ("same_again", {}),
        ("position_shifted", {"position": (1 << 17) + 4096}),
        ("longer_signal", {"length": 1 << 19}),
        ("amplitude_0.25", {"amplitude": 0.25}),
    ]:
        other = impulse_response(**kwargs)
        diff = max(abs(base.get(d, 0.0) - other.get(d, 0.0)) for d in range(lo, lo + len(reference)))
        report["reproducibility"].append({"case": label, "max_abs_diff": diff})

    # 2. 実際の入力と soxr_hq の一致、タップ数を減らしたときの誤差
    inputs = {}
    for name in INPUT_IDS:
        path = INPUT_DIR / f"{name}_src_tk_1.wav"
        x, rate = sf.read(path, dtype="float32")
        if rate != SOURCE_RATE or x.ndim != 1:
            raise SystemExit(f"{path} は 44.1 kHz モノラルではない")
        ref_wave, _ = librosa.load(path, sr=TARGET_RATE)
        inputs[name] = (x, ref_wave, normalized_mel(ref_wave))

    eval_hz = np.linspace(0, 10000, 2001)
    w = 2 * np.pi * eval_hz / SOURCE_RATE
    ref_response = signal.freqz(reference, worN=w)[1]

    for threshold in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        tlo, c = trim(base, threshold)
        c32 = c.astype(np.float32).astype(np.float64)
        wave_err, mel_max, mel_mae = 0.0, 0.0, 0.0
        for x, ref_wave, ref_mel in inputs.values():
            y = decimate(x, tlo, c32)
            wave_err = max(wave_err, float(np.abs(y - ref_wave).max()))
            diff = np.abs(normalized_mel(y) - ref_mel)
            mel_max = max(mel_max, float(diff.max()))
            mel_mae = max(mel_mae, float(diff.mean()))
        response = signal.freqz(c, worN=w)[1]
        report["truncation"].append({
            "threshold": threshold, "taps": int(len(c)), "d_first": tlo, "d_last": tlo + len(c) - 1,
            "wave_max_abs_err": wave_err, "mel_max_abs_err": mel_max, "mel_worst_input_mae": mel_mae,
            "passband_response_max_db_dev_0_10kHz": float(
                np.abs(20 * np.log10(np.maximum(np.abs(response), 1e-12) / np.maximum(np.abs(ref_response), 1e-12))).max()
            ),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def generate(threshold: float) -> int:
    lo, c = trim(impulse_response(), threshold)
    c32 = c.astype(np.float32)
    digest = hashlib.sha256(c32.tobytes()).hexdigest()
    v = versions()
    rows = []
    for start in range(0, len(c32), 4):
        rows.append("        " + ", ".join(float(value).hex() for value in c32[start:start + 4]) + ",")
    body = "\n".join(rows)
    SWIFT_PATH.write_text(f"""// このファイルは scripts/generate_resampler_fir.py が生成する。手で編集しない。
//
// 生成時の環境: soxr {v['soxr']} / libsoxr {v['libsoxr']} / librosa {v['librosa']} / numpy {v['numpy']}
// 再生成: PronounSE/venv/bin/python scripts/generate_resampler_fir.py generate --threshold {threshold:g}

/// soxr_hq の 44.1 kHz → 22.05 kHz のインパルス応答（Float32）
///
/// y[n] = Σ_j coefficients[j] · x[2n - firstOffset - j]
enum SoxrHQDecimatorCoefficients {{
    /// 係数の先頭に対応する d（= 2n - p）
    static let firstOffset = {lo}
    /// 係数を切り出したしきい値（|h| がこの値を超える範囲を残した）
    static let threshold: Double = {threshold:g}
    /// Float32 の係数のバイト列の SHA-256
    static let sha256 = "{digest}"

    static let coefficients: [Float] = [
{body}
    ]
}}
""")
    print(f"taps={len(c32)} firstOffset={lo} sha256={digest}")
    print(f"wrote {SWIFT_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--out", type=Path, default=REPO_ROOT / "metrics/preprocess_fir/analysis.json")
    g = sub.add_parser("generate")
    g.add_argument("--threshold", type=float, default=1e-7)
    args = parser.parse_args()
    return analyze(args.out) if args.command == "analyze" else generate(args.threshold)


if __name__ == "__main__":
    sys.exit(main())
