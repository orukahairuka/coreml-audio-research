#!/usr/bin/env python3
"""Swift の前処理の各段階を、公開実装（PronounSE の get_spectrograms）と比べる。

入力は preprocess モード（実機）または --build-mac（Mac）が保存した
preprocess_<legacy|current>_in<入力>_<段階>.npy。preprocess_runs.csv があれば時間とメモリも集計する。

実行例:
  # Mac でビルドして比較する
  PronounSE/venv/bin/python scripts/compare_preprocess_parity.py --build-mac metrics/preprocess_parity_<日時>/mac
  # 実機の結果と比較する
  PronounSE/venv/bin/python scripts/compare_preprocess_parity.py metrics/preprocess_parity_<日時>/iPhone_13
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import librosa  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "PronounSE/Transformer"))
from utils import get_spectrograms, preprocess  # noqa: E402

INPUTS = {"1": "入力1", "6": "入力2", "11": "入力3", "2": "入力4"}
INPUT_DIR = REPO_ROOT / "data/thesis_ch4/inputs"
TARGET_T = 262
AUDIO_DIR = REPO_ROOT / "ios/CoreMLAudioApp/CoreMLAudioApp/Models/Audio"
MAC_SOURCES = ["AudioFeatureExtractor", "LegacyAudioFeatureExtractor", "SoxrHQDecimator",
               "SoxrHQDecimatorCoefficients", "NpyWriter", "PreprocessStages"]


def python_stages(name: str) -> dict[str, np.ndarray]:
    path = INPUT_DIR / f"{name}_src_tk_1.wav"
    y, _ = librosa.load(path, sr=22050)
    load = y.copy()
    prep = preprocess(y.copy())
    emph = np.append(prep[0], prep[1:] - 0.97 * prep[:-1])
    mag = np.abs(librosa.stft(y=emph, n_fft=1024, hop_length=256, win_length=1024))
    mel = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=256) @ mag
    norm = np.clip((20 * np.log10(np.maximum(1e-5, mel)) - 20 + 100) / 100, 1e-8, 1).T.astype(np.float32)
    reference, _ = get_spectrograms(str(path))
    if not np.array_equal(norm, reference):
        raise SystemExit(f"{name}: 段階ごとの計算が get_spectrograms と一致しない")
    return {"load": load, "prep": prep, "emph": emph, "mag": mag.T, "mel": mel.T, "norm": norm}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = x - y
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return {"max_abs": float(np.abs(d).max()), "mae": float(np.abs(d).mean()), "signed": float(d.mean()),
            "cos": float(x @ y / denominator) if denominator else None}


def value_range(a: np.ndarray) -> dict:
    return {"min": float(a.min()), "max": float(a.max()),
            "lower_pct": float((a <= 1e-8).mean() * 100), "upper_pct": float((a >= 1).mean() * 100)}


def parity(root: Path) -> dict:
    report: dict = {}
    for name, label in INPUTS.items():
        reference = python_stages(name)
        entry: dict = {"T_python": int(len(reference["norm"]))}
        for implementation in ["legacy", "current"]:
            files = {f.name: f for f in root.rglob(f"preprocess_{implementation}_in{name}_*.npy")}
            result: dict = {}
            for stage, ref in reference.items():
                path = files.get(f"preprocess_{implementation}_in{name}_{stage}.npy")
                if path is None:
                    raise SystemExit(f"{implementation} {name} {stage} の npy が無い: {root}")
                value = np.load(path)
                value = value[0] if stage in {"load", "prep", "emph"} else value
                count = min(len(value), len(ref))
                result[stage] = {"length": int(len(value)), "length_python": int(len(ref)),
                                 **compare(value[:count], ref[:count])}
            norm = np.load(files[f"preprocess_{implementation}_in{name}_norm.npy"])
            t_prime = min(len(norm), TARGET_T)
            count = min(t_prime, len(reference["norm"]))
            result["model_input"] = {
                "T": int(len(norm)), "T_prime": int(t_prime),
                **compare(norm[:count], reference["norm"][:count]),
                "range": value_range(norm[:count]), "range_python": value_range(reference["norm"][:count]),
            }
            entry[implementation] = result
        report[label] = entry
    return report


def timing(root: Path) -> dict:
    rows = [row for path in root.rglob("preprocess_runs.csv") for row in csv.DictReader(open(path, newline=""))]
    groups: dict = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            groups[(INPUTS.get(row["input_name"], row["input_name"]), row["implementation"], row["phase"])].append(row)
    out: dict = {"failed_runs": sum(row["status"] != "success" for row in rows),
                 "thermal": sorted({row["thermal_after"] for row in rows}), "by_input": {}}
    for (label, implementation, phase), items in sorted(groups.items()):
        if phase == "warmup":
            continue
        target = out["by_input"].setdefault(label, {}).setdefault(implementation, {})
        if phase == "steady":
            target["n"] = len(items)
            target["T"] = sorted({int(r["T"]) for r in items})
            target["T_prime"] = sorted({int(r["T_prime"]) for r in items})
            target["mel_sha256_distinct"] = len({r["mel_sha256"] for r in items})
            for key in ["read_ms", "resample_ms", "load_ms", "mel_ms", "total_ms"]:
                values = [float(r[key]) for r in items if r[key]]
                if values:
                    target[f"{key}_median"] = st.median(values)
        else:
            target["memory_runs"] = len(items)
            target["heap_delta_mb_median"] = st.median(int(r["heap_delta_bytes"]) / 1e6 for r in items)
            target["heap_delta_mb_max"] = max(int(r["heap_delta_bytes"]) / 1e6 for r in items)
    return out


def build_mac(output: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "preprocess_parity"
        sources = [str(REPO_ROOT / "scripts/preprocess_parity/main.swift")] + [str(AUDIO_DIR / f"{n}.swift") for n in MAC_SOURCES]
        subprocess.run(["swiftc", "-O", "-swift-version", "5", "-o", str(binary), *sources],
                       check=True, stdin=subprocess.DEVNULL, capture_output=True)
        subprocess.run([str(binary), str(output), *[str(INPUT_DIR / f"{n}_src_tk_1.wav") for n in INPUTS]],
                       check=True, stdin=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--build-mac", action="store_true")
    args = parser.parse_args()
    if args.build_mac:
        build_mac(args.root)
    report = {"parity": parity(args.root), "timing": timing(args.root)}
    (args.root / "parity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for label, entry in report["parity"].items():
        for implementation in ["legacy", "current"]:
            m = entry[implementation]["model_input"]
            print(f"{label} {implementation:7s} T={m['T']}/{entry['T_python']} T'={m['T_prime']} max={m['max_abs']:.3e} "
                  f"mae={m['mae']:.3e} signed={m['signed']:+.3e} cos={m['cos']:.8f} "
                  f"range=[{m['range']['min']:.3g},{m['range']['max']:.4f}] lower={m['range']['lower_pct']:.2f}% "
                  f"upper={m['range']['upper_pct']:.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
