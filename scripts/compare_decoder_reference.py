#!/usr/bin/env python3
"""Phase 3 — PyTorch Decoder 参照値と iOS CoreML run を比較する。

`generate_decoder_reference.py` が書いた reference dir と、iOS の debug run dir を
入力に取り、以下を比較する:

1. **最終 postnet_output**: iOS の `postnet_output.npy` vs PyTorch の `postnet_output_final.npy`
   - max_abs_diff / allclose(1e-2 / 1e-3 / 1e-4)
2. **per-step mel 統計**: iOS の `decoder_steps.csv` vs PyTorch の `decoder_step_stats.json`
   - 初めて mel_min / max / mean が乖離する step を発火点候補として出力
3. **NaN/Inf 発火点**: iOS の decoder_steps.csv で NaN/Inf が出る最初の step

計画書: `docs/2026-05-19/all-engine-precision-stability-plan.md` §5.2

使い方:
    scripts/compare_decoder_reference.py \\
        --reference data/2026-05-19/decoder_reference/<tag> \\
        --ios-run audio/<archive>/debug/<runId> \\
        --md docs/2026-05-19/decoder-internal-divergence.md

複数 iOS run を指定したい場合は `--ios-run` を複数回付ける。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_postnet(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim == 2:
        # iOS NpyWriter は [T, n_mels] 2D で書く。reference は [1, T, n_mels] 3D。
        arr = arr[np.newaxis, :, :]
    return arr.astype(np.float32)


def tensor_diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        # broadcast 失敗するなら shape を合わせて報告のみ
        return {
            "shape_a": tuple(a.shape),
            "shape_b": tuple(b.shape),
            "diffable": False,
        }
    diff = np.abs(a - b)
    return {
        "shape": tuple(a.shape),
        "max_abs_diff": float(np.nanmax(diff)),
        "mean_abs_diff": float(np.nanmean(diff)),
        "allclose_1e-2": bool(np.allclose(a, b, atol=1e-2)),
        "allclose_1e-3": bool(np.allclose(a, b, atol=1e-3)),
        "allclose_1e-4": bool(np.allclose(a, b, atol=1e-4)),
        "ref_max_abs": float(np.max(np.abs(b))),
        "actual_max_abs": float(np.max(np.abs(a))),
        "diffable": True,
    }


def read_ios_decoder_steps(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                out.append({
                    "step": int(row["step"]),
                    "mel_min": _try_float(row["mel_min"]),
                    "mel_max": _try_float(row["mel_max"]),
                    "mel_mean": _try_float(row["mel_mean"]),
                    "mel_sha": row["mel_sha"],
                })
            except (KeyError, ValueError):
                continue
    return out


def _try_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        if s.lower() == "nan":
            return float("nan")
        if s.lower() == "inf":
            return float("inf")
        if s.lower() == "-inf":
            return float("-inf")
        return float("nan")


def find_first_divergence(
    ref_stats: list[dict],
    ios_stats: list[dict],
    atol_mean: float = 1e-2,
    atol_minmax: float = 1e-1,
) -> dict | None:
    """per-step mel 統計の最初の乖離 step を探す。"""
    by_step_ref = {s["step"]: s for s in ref_stats}
    for ios in ios_stats:
        step = ios["step"]
        ref = by_step_ref.get(step)
        if ref is None:
            continue
        # NaN/Inf を ref 側と突き合わせる前に、ios 側だけ壊れているケースを優先で報告
        if math.isnan(ios["mel_mean"]) or math.isinf(ios["mel_mean"]) \
           or math.isnan(ios["mel_min"]) or math.isinf(ios["mel_min"]) \
           or math.isnan(ios["mel_max"]) or math.isinf(ios["mel_max"]):
            return {
                "step": step,
                "kind": "nan_inf_ios_side",
                "ios": ios,
                "ref": ref,
            }
        if abs(ios["mel_mean"] - ref["mel_mean"]) > atol_mean \
           or abs(ios["mel_min"] - ref["mel_min"]) > atol_minmax \
           or abs(ios["mel_max"] - ref["mel_max"]) > atol_minmax:
            return {
                "step": step,
                "kind": "stat_drift",
                "ios": ios,
                "ref": ref,
                "mean_delta": ios["mel_mean"] - ref["mel_mean"],
                "min_delta": ios["mel_min"] - ref["mel_min"],
                "max_delta": ios["mel_max"] - ref["mel_max"],
            }
    return None


def find_first_nan_inf(ios_stats: list[dict]) -> dict | None:
    for s in ios_stats:
        if any(math.isnan(s[k]) or math.isinf(s[k]) for k in ("mel_min", "mel_max", "mel_mean")):
            return s
    return None


def compare_run(reference_dir: Path, ios_run: Path) -> dict:
    label = ios_run.name
    result: dict = {"label": label, "ios_run": str(ios_run)}

    # 1. postnet 比較
    ref_post = load_postnet(reference_dir / "postnet_output_final.npy")
    ios_post = load_postnet(ios_run / "postnet_output.npy")
    if ref_post is None or ios_post is None:
        result["postnet_diff"] = {"diffable": False, "reason": "missing file"}
    else:
        result["postnet_diff"] = tensor_diff_stats(ios_post, ref_post)

    # 2. per-step mel 比較
    ref_steps_path = reference_dir / "decoder_step_stats.json"
    if ref_steps_path.exists():
        ref_steps = json.loads(ref_steps_path.read_text())
    else:
        ref_steps = []
    ios_steps = read_ios_decoder_steps(ios_run / "decoder_steps.csv")
    result["ref_step_count"] = len(ref_steps)
    result["ios_step_count"] = len(ios_steps)
    result["first_divergence"] = find_first_divergence(ref_steps, ios_steps)
    result["first_nan_inf_ios"] = find_first_nan_inf(ios_steps)

    return result


def format_postnet_section(results: list[dict]) -> list[str]:
    lines = []
    lines.append("## 1. 最終 postnet_output の diff")
    lines.append("")
    lines.append("| run | shape | max_abs_diff | mean_abs_diff | allclose 1e-2 | 1e-3 | 1e-4 | ref max | actual max |")
    lines.append("|---|---|---:|---:|---|---|---|---:|---:|")
    for r in results:
        d = r.get("postnet_diff", {})
        if not d.get("diffable"):
            lines.append(f"| {r['label']} | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {r['label']} | {d.get('shape','-')} | {d.get('max_abs_diff', float('nan')):.4e} | "
            f"{d.get('mean_abs_diff', float('nan')):.4e} | "
            f"{'✅' if d.get('allclose_1e-2') else '❌'} | "
            f"{'✅' if d.get('allclose_1e-3') else '❌'} | "
            f"{'✅' if d.get('allclose_1e-4') else '❌'} | "
            f"{d.get('ref_max_abs', float('nan')):.4f} | "
            f"{d.get('actual_max_abs', float('nan')):.4f} |"
        )
    lines.append("")
    return lines


def format_divergence_section(results: list[dict]) -> list[str]:
    lines = []
    lines.append("## 2. per-step mel 統計の最初の乖離 step")
    lines.append("")
    lines.append("| run | step | kind | ios mean | ref mean | Δmean | ios min | ios max |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in results:
        d = r.get("first_divergence")
        if d is None:
            lines.append(f"| {r['label']} | - | (no divergence) | - | - | - | - | - |")
            continue
        ios = d["ios"]
        ref = d["ref"]
        delta = d.get("mean_delta")
        delta_str = "-" if delta is None else f"{delta:+.4e}"
        lines.append(
            f"| {r['label']} | {d['step']} | {d['kind']} | "
            f"{ios['mel_mean']:.4e} | {ref['mel_mean']:.4e} | {delta_str} | "
            f"{ios['mel_min']:.4e} | {ios['mel_max']:.4e} |"
        )
    lines.append("")
    return lines


def format_nan_section(results: list[dict]) -> list[str]:
    lines = []
    lines.append("## 3. iOS 側で NaN/Inf が初発する step")
    lines.append("")
    lines.append("| run | step | mel_min | mel_max | mel_mean |")
    lines.append("|---|---:|---|---|---|")
    for r in results:
        s = r.get("first_nan_inf_ios")
        if s is None:
            lines.append(f"| {r['label']} | - | - | - | - |")
            continue
        lines.append(
            f"| {r['label']} | {s['step']} | {s['mel_min']} | {s['mel_max']} | {s['mel_mean']} |"
        )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="PyTorch decoder reference dir (generate_decoder_reference.py の出力)")
    parser.add_argument("--ios-run", action="append", required=True, help="iOS debug run dir。複数指定可")
    parser.add_argument("--md", default="docs/2026-05-19/decoder-internal-divergence.md")
    args = parser.parse_args()

    reference_dir = Path(args.reference)
    if not reference_dir.is_dir():
        print(f"# reference not found: {reference_dir}", file=sys.stderr)
        return 1

    results = []
    for ios_run_str in args.ios_run:
        ios_run = Path(ios_run_str)
        if not ios_run.is_dir():
            print(f"# ios run not found: {ios_run}", file=sys.stderr)
            continue
        results.append(compare_run(reference_dir, ios_run))

    if not results:
        print("# no comparable runs", file=sys.stderr)
        return 1

    md_lines: list[str] = []
    md_lines.append("# Phase 3 — Decoder 内部 divergence point")
    md_lines.append("")
    md_lines.append(f"PyTorch reference: `{reference_dir}`")
    md_lines.append("")
    md_lines.append(f"比較対象 iOS run: {len(results)} 件")
    md_lines.append("")
    md_lines.append("生成元スクリプト: `scripts/compare_decoder_reference.py`")
    md_lines.append("")
    md_lines.extend(format_postnet_section(results))
    md_lines.extend(format_divergence_section(results))
    md_lines.extend(format_nan_section(results))

    md_lines.append("## 4. 次に見るべき layer")
    md_lines.append("")
    md_lines.append("PyTorch reference の `step_001/*.npy` には各境界の中間 tensor が保存されている。")
    md_lines.append("iOS 多出力 CoreML モデル（未実装）で同等の中間 tensor を取れれば、")
    md_lines.append("ここで列挙した divergence step の中で、prenet / norm / selfattn_i / dotattn_i / ffn_i /")
    md_lines.append("mel_linear / postconvnet のどの境界が原因かを切り分けられる。")
    md_lines.append("")

    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines))
    print(f"# wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
