#!/usr/bin/env python3
"""Phase 1 — 全 12 組合せ安定性マトリクス集計スクリプト。

`Documents/Result/debug/` を吸い出した `audio/<archive>/debug/<runId>/` 群を歩いて、
各 run の precision / computeUnit / iter / wav rms / peak / NaN/Inf / 各段 sha256 /
Decoder step1 stats を集計し、CSV と markdown を吐く。

入力:
    1 つ以上の `audio/<archive>/` ディレクトリ（または直接 `debug/` を指定）。
    glob パターン可。

出力:
    --csv で指定したファイル（既定: `data/2026-05-19/stability_matrix.csv`）
    --md で指定したファイル（既定: `docs/2026-05-19/stability-matrix-results.md`）

使い方:
    scripts/aggregate_stability_matrix.py audio/iPhone_*_phase1_*

各 run の `summary.txt` を読み、ラベル末尾の数字を iter として抜く
（例: `directRepeat1` → iter=1）。precision/computeUnit は summary.txt の行から取得。

decoder_steps.csv があれば step1 (= 自己回帰 0 始まりの step 1) の mel_min/max/mean を採る。
NaN/Inf はその step1 の値が `nan` / `inf` / `-inf` かどうかで判定する。
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


STAGES = (
    "mel_normalized",
    "encoder_output",
    "postnet_output",
    "waveform_predeemph",
    "waveform_postdeemph",
)


def parse_summary(path: Path) -> dict:
    info: dict = {}
    if not path.exists():
        return info
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        for stage in STAGES:
            if line.startswith(stage + ":"):
                m = re.search(r"sha256=([0-9a-f]+)", line)
                shape_m = re.search(r"shape=(\[[^\]]+\])", line)
                rms_m = re.search(r"rms=([-+0-9.eE]+|nan|inf|-inf)", line)
                if m:
                    info[stage + "_sha"] = m.group(1)[:10]
                if shape_m:
                    info[stage + "_shape"] = shape_m.group(1)
                if rms_m:
                    info[stage + "_rms"] = rms_m.group(1)
        if line.startswith("label="):
            # label=<text> pid=<n> underXCTestRunner=<bool>
            m = re.search(r"label=(\S+)", line)
            if m:
                info["label"] = m.group(1)
        if line.startswith("precision="):
            # precision=<p> computeUnit=<cu> shapeMode=<sm>
            m_p = re.search(r"precision=(\S+)", line)
            m_cu = re.search(r"computeUnit=(\S+)", line)
            m_sm = re.search(r"shapeMode=(\S+)", line)
            if m_p:
                info["precision"] = m_p.group(1)
            if m_cu:
                info["computeUnit"] = m_cu.group(1)
            if m_sm:
                info["shapeMode"] = m_sm.group(1)
        if line.startswith("runId="):
            info["runId"] = line.split("=", 1)[1]
    return info


def parse_decoder_step(path: Path, target_step: int = 1) -> dict:
    """decoder_steps.csv の指定 step 行を辞書で返す。"""
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["step"]) == target_step:
                    return {
                        f"decoder_step{target_step}_mel_min": row["mel_min"],
                        f"decoder_step{target_step}_mel_max": row["mel_max"],
                        f"decoder_step{target_step}_mel_mean": row["mel_mean"],
                        f"decoder_step{target_step}_mel_sha": row["mel_sha"],
                    }
    except (KeyError, ValueError, OSError):
        return {}
    return {}


def read_npy_postdeemph(run_dir: Path) -> Optional[list[float]]:
    """`waveform_postdeemph.npy` (1, N float32) を読む。NpyWriter が書いたバイナリ前提。"""
    path = run_dir / "waveform_postdeemph.npy"
    if not path.exists():
        path = run_dir / "waveform.npy"  # legacy 命名フォールバック
    if not path.exists():
        return None
    try:
        return read_simple_npy_float32(path)
    except (OSError, ValueError):
        return None


def read_simple_npy_float32(path: Path) -> list[float]:
    """NpyWriter.writeFloat32 が吐く NumPy v1 形式 (little-endian, fortran=False) を最低限読む。"""
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"not a npy file: {path}")
        major = f.read(1)[0]
        _minor = f.read(1)[0]
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        else:
            header_len = struct.unpack("<I", f.read(4))[0]
        header = f.read(header_len).decode("latin-1")
        # 例: {'descr': '<f4', 'fortran_order': False, 'shape': (1, 65536), }
        if "<f4" not in header:
            raise ValueError(f"unsupported dtype in {path}: {header.strip()}")
        body = f.read()
        n = len(body) // 4
        return list(struct.unpack("<" + "f" * n, body))


def wav_stats_from_samples(samples: list[float]) -> tuple[float, float, bool, bool]:
    """float サンプルから int16 換算の (rms, peak, hasNaN, hasInf) を返す。"""
    has_nan = False
    has_inf = False
    sum_sq = 0.0
    peak = 0.0
    count = 0
    for v in samples:
        if math.isnan(v):
            has_nan = True
            continue
        if math.isinf(v):
            has_inf = True
            continue
        count += 1
        sum_sq += v * v
        if abs(v) > peak:
            peak = abs(v)
    if count == 0:
        return (float("nan"), float("nan"), has_nan, has_inf)
    rms = math.sqrt(sum_sq / count)
    return (rms * 32767.0, peak * 32767.0, has_nan, has_inf)


def classify(rms_i16: float, peak_i16: float, has_nan: bool, has_inf: bool) -> str:
    """Phase 1 計画書 §2.4 と同じルール。"""
    if has_nan or has_inf:
        return "nan_inf"
    if math.isnan(rms_i16):
        return "predict_failed"
    if peak_i16 > 32000 and rms_i16 > 7000:
        return "clipped"
    if rms_i16 < 3000:
        return "quiet"
    return "normal_loud"


def extract_iter_from_label(label: str) -> Optional[int]:
    m = re.search(r"(\d+)$", label or "")
    if m:
        return int(m.group(1))
    return None


def walk_debug_runs(roots: list[Path]) -> list[Path]:
    runs: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        # 入力が `audio/<archive>/` か `debug/` か `debug/<runId>/` かを吸収する
        candidates = [root]
        if (root / "debug").is_dir():
            candidates.append(root / "debug")
        for c in candidates:
            for entry in sorted(c.glob("*")):
                if (entry / "summary.txt").exists():
                    runs.append(entry)
                elif entry.is_dir() and entry.name == "debug":
                    for sub in sorted(entry.iterdir()):
                        if (sub / "summary.txt").exists():
                            runs.append(sub)
    # 重複除去（同 path）
    seen = set()
    unique = []
    for r in runs:
        key = r.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="audio/<dir>/ または debug/<runId>/ のパス（glob 可）")
    parser.add_argument("--csv", default="data/2026-05-19/stability_matrix.csv")
    parser.add_argument("--md", default="docs/2026-05-19/stability-matrix-results.md")
    args = parser.parse_args()

    expanded: list[Path] = []
    for pattern in args.inputs:
        matches = [Path(p) for p in glob.glob(pattern)]
        if not matches:
            print(f"# warn: pattern matched nothing: {pattern}", file=sys.stderr)
        expanded.extend(matches)

    runs = walk_debug_runs(expanded)
    if not runs:
        print("# no debug runs found", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for run_dir in runs:
        info = parse_summary(run_dir / "summary.txt")
        info.update(parse_decoder_step(run_dir / "decoder_steps.csv", target_step=1))

        samples = read_npy_postdeemph(run_dir)
        if samples is not None:
            rms_i16, peak_i16, has_nan, has_inf = wav_stats_from_samples(samples)
        else:
            rms_i16 = float("nan")
            peak_i16 = float("nan")
            has_nan = False
            has_inf = False

        info["wav_rms_int16"] = "" if math.isnan(rms_i16) else f"{rms_i16:.1f}"
        info["wav_peak_int16"] = "" if math.isnan(peak_i16) else f"{peak_i16:.1f}"
        info["has_nan"] = "true" if has_nan else "false"
        info["has_inf"] = "true" if has_inf else "false"
        info["classification"] = classify(rms_i16, peak_i16, has_nan, has_inf)
        info["iter"] = extract_iter_from_label(info.get("label", ""))
        info["run_dir"] = str(run_dir)
        rows.append(info)

    # CSV 出力
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "precision", "computeUnit", "shapeMode", "label", "iter", "classification",
        "wav_rms_int16", "wav_peak_int16", "has_nan", "has_inf",
        "mel_normalized_sha", "encoder_output_sha", "postnet_output_sha", "waveform_postdeemph_sha",
        "decoder_step1_mel_min", "decoder_step1_mel_max", "decoder_step1_mel_mean", "decoder_step1_mel_sha",
        "runId", "run_dir",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"# wrote {csv_path} ({len(rows)} rows)")

    # markdown 集計
    md_lines: list[str] = []
    md_lines.append("# Phase 1 — 全 12 組合せ安定性マトリクス結果")
    md_lines.append("")
    md_lines.append(f"集計対象: {len(rows)} runs, 取り込み元 {len(expanded)} archive")
    md_lines.append("")
    md_lines.append("生成元スクリプト: `scripts/aggregate_stability_matrix.py`")
    md_lines.append("")

    # 分類カウント（precision × computeUnit × iter ピボット）
    pivot: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r.get("precision") or "?", r.get("computeUnit") or "?")
        pivot[key][r.get("iter")].append(r["classification"])

    md_lines.append("## 1. 分類別カウント（precision × computeUnit）")
    md_lines.append("")
    md_lines.append("| precision | computeUnit | normal_loud | quiet | clipped | nan_inf | predict_failed |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (prec, cu) in sorted(pivot.keys()):
        all_cls: list[str] = []
        for cls_list in pivot[(prec, cu)].values():
            all_cls.extend(cls_list)
        c = {k: all_cls.count(k) for k in ("normal_loud", "quiet", "clipped", "nan_inf", "predict_failed")}
        md_lines.append(f"| {prec} | {cu} | {c['normal_loud']} | {c['quiet']} | {c['clipped']} | {c['nan_inf']} | {c['predict_failed']} |")
    md_lines.append("")

    md_lines.append("## 2. iter 別ヒートマップ（precision × computeUnit × iter）")
    md_lines.append("")
    md_lines.append("値は最多分類を示す。複数同数なら `?`。")
    md_lines.append("")
    iters_seen = sorted({r.get("iter") for r in rows if r.get("iter") is not None})
    if iters_seen:
        header = "| precision | computeUnit | " + " | ".join(f"iter{i}" for i in iters_seen) + " |"
        sep = "|---|---|" + "|".join(["---"] * len(iters_seen)) + "|"
        md_lines.append(header)
        md_lines.append(sep)
        for (prec, cu) in sorted(pivot.keys()):
            cells = []
            for i in iters_seen:
                cls_list = pivot[(prec, cu)].get(i, [])
                if not cls_list:
                    cells.append("-")
                    continue
                counts = defaultdict(int)
                for c in cls_list:
                    counts[c] += 1
                top = max(counts.values())
                top_classes = [k for k, v in counts.items() if v == top]
                cells.append(top_classes[0] if len(top_classes) == 1 else "?")
            md_lines.append(f"| {prec} | {cu} | " + " | ".join(cells) + " |")
        md_lines.append("")

    md_lines.append("## 3. NaN/Inf を含む run の Decoder step1 統計")
    md_lines.append("")
    nan_rows = [r for r in rows if r["classification"] == "nan_inf"]
    if nan_rows:
        md_lines.append("| run | iter | step1 mel_min | step1 mel_max | step1 mel_mean | postnet_sha |")
        md_lines.append("|---|---|---|---|---|---|")
        for r in nan_rows:
            md_lines.append(
                f"| {r.get('label','?')} | {r.get('iter','?')} | "
                f"{r.get('decoder_step1_mel_min','-')} | {r.get('decoder_step1_mel_max','-')} | "
                f"{r.get('decoder_step1_mel_mean','-')} | {r.get('postnet_output_sha','-')} |"
            )
    else:
        md_lines.append("（NaN/Inf を含む run はなし）")
    md_lines.append("")

    md_lines.append("## 4. 個別 run 一覧")
    md_lines.append("")
    md_lines.append("| precision | computeUnit | iter | class | rms | peak | postnet sha | run |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(
        rows,
        key=lambda x: (x.get("precision") or "", x.get("computeUnit") or "", x.get("iter") or 0),
    ):
        md_lines.append(
            f"| {r.get('precision','?')} | {r.get('computeUnit','?')} | {r.get('iter','?')} | "
            f"{r['classification']} | {r.get('wav_rms_int16','-')} | {r.get('wav_peak_int16','-')} | "
            f"{r.get('postnet_output_sha','-')} | {r.get('label','?')} |"
        )
    md_lines.append("")

    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines))
    print(f"# wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
