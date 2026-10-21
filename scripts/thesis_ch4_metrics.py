#!/usr/bin/env python3
"""卒論第4章：実機の CSV の各行に、npy / wav から計算する比較の指標を足す。

run_thesis_bench.py が吸い出した raw/device/ を読み、次の列を足した CSV を raw/ の隣に書く。

infer_runs.csv → infer_runs_metrics.csv
  基準は同じ入力の Float32 + cpuOnly の、最初の定常時の実行（phase=steady の run_index 最小）
  - hifigan_rms_ratio, hifigan_peak_ratio, hifigan_cosine   … HiFi-GAN の出力
  - deemph_rms_ratio, deemph_peak_ratio, deemph_cosine      … デエンファシスの後
  - decoder_mel_cosine                                      … Decoder の出力メル
  - wav_saturation_ratio                                    … 保存した WAV の値が 32767 / -32768 の割合
vocoder_runs.csv → vocoder_runs_metrics.csv
  基準は同じメルを入れた PyTorch 版 HiFi-GAN（data/thesis_ch4/pytorch_reference/<id>/）
  列は上と同じ（decoder_mel_cosine は無し）

指標に閾値は設けない（正常・異常の分類はしない）。

実行例:
    PronounSE/venv/bin/python scripts/thesis_ch4_metrics.py metrics/thesis_ch4_iPhone_3_<日時>/raw
"""

from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTORCH_REF = REPO_ROOT / "data/thesis_ch4/pytorch_reference"


def load_npy(path: Path) -> np.ndarray | None:
    return np.load(path).astype(np.float64).reshape(-1) if path.is_file() else None


def compare(ref: np.ndarray | None, cur: np.ndarray | None) -> dict:
    """RMS 比・Peak 比・コサイン類似度。長さが違うときは短い方に揃える。NaN/Inf はそのまま伝わる"""
    if ref is None or cur is None:
        return {"rms_ratio": np.nan, "peak_ratio": np.nan, "cosine": np.nan}
    n = min(ref.size, cur.size)
    a, b = ref[:n], cur[:n]
    ref_rms = np.sqrt(np.mean(a * a))
    cur_rms = np.sqrt(np.mean(b * b))
    ref_peak = np.max(np.abs(a)) if n else np.nan
    cur_peak = np.max(np.abs(b)) if n else np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "rms_ratio": float(cur_rms / ref_rms) if ref_rms > 0 else np.nan,
            "peak_ratio": float(cur_peak / ref_peak) if ref_peak > 0 else np.nan,
            "cosine": float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan,
        }


def wav_saturation_ratio(path: Path) -> float:
    """16 ビット WAV のうち、値が 32767 または -32768 のサンプルの割合"""
    if not path.is_file():
        return np.nan
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            return np.nan
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if data.size == 0:
        return np.nan
    return float(np.mean((data == 32767) | (data == -32768)))


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def add_metrics(rows: list[dict], device_dir: Path, ref_for_row, with_decoder_mel: bool) -> list[str]:
    """各行に指標の列を足し、足した列名を返す"""
    added: list[str] = []
    for r in rows:
        out = {}
        prefix = r.get("file_prefix", "")
        if r.get("status") == "success" and prefix:
            base = device_dir / prefix
            ref = ref_for_row(r)
            for stage in ("hifigan", "deemph"):
                m = compare(ref.get(stage), load_npy(Path(f"{base}_{stage}.npy")))
                for k, v in m.items():
                    out[f"{stage}_{k}"] = v
            if with_decoder_mel:
                out["decoder_mel_cosine"] = compare(
                    ref.get("decoder_mel"), load_npy(Path(f"{base}_decoder_mel.npy")))["cosine"]
            out["wav_saturation_ratio"] = wav_saturation_ratio(Path(f"{base}_deemph.wav"))
            out["reference"] = ref.get("ref_prefix", "")
        for k, v in out.items():
            r[k] = v
            if k not in added:
                added.append(k)
    return added


def infer_reference(rows: list[dict], device_dir: Path):
    """入力ごとに、Float32 + cpuOnly の最初の定常時の実行の npy を基準にする"""
    base = [r for r in rows if r["precision"] == "Float32" and r["compute_units"] == "cpuOnly"
            and r["phase"] == "steady" and r["status"] == "success"]
    base.sort(key=lambda r: (r["launch_id"], int(r["run_index"])))
    refs: dict[str, dict] = {}
    for r in base:
        name = r["input_name"]
        if name in refs:
            continue
        p = device_dir / r["file_prefix"]
        refs[name] = {
            "hifigan": load_npy(Path(f"{p}_hifigan.npy")),
            "deemph": load_npy(Path(f"{p}_deemph.npy")),
            "decoder_mel": load_npy(Path(f"{p}_decoder_mel.npy")),
            "ref_prefix": r["file_prefix"],
        }
    return lambda r: refs.get(r["input_name"], {})


def vocoder_reference(pytorch_ref: Path):
    cache: dict[str, dict] = {}

    def get(r):
        name = r["input_name"]
        if name not in cache:
            d = pytorch_ref / name
            cache[name] = {
                "hifigan": load_npy(d / "waveform_predeemph.npy"),
                "deemph": load_npy(d / "waveform_postdeemph.npy"),
                "ref_prefix": f"pytorch/{name}",
            }
        return cache[name]
    return get


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_dir", type=Path, help="run_thesis_bench.py の出力（raw/）")
    parser.add_argument("--pytorch-ref", type=Path, default=DEFAULT_PYTORCH_REF)
    args = parser.parse_args()

    device_dir = args.raw_dir / "device"
    for name, ref_fn, with_mel in (
        ("infer_runs", None, True),
        ("vocoder_runs", vocoder_reference(args.pytorch_ref), False),
    ):
        src = device_dir / f"{name}.csv"
        if not src.is_file():
            print(f"# skip: {src} がない")
            continue
        header, rows = read_rows(src)
        fn = ref_fn or infer_reference(rows, device_dir)
        added = add_metrics(rows, device_dir, fn, with_mel)
        dest = args.raw_dir / f"{name}_metrics.csv"
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header + added)
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {dest} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
