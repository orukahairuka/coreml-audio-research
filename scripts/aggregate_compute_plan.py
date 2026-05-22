#!/usr/bin/env python3
"""Phase 4 — MLComputePlan の per-op device 配置 JSON をまとめて markdown にする。

入力: `Documents/Result/debug/compute_plan/<precision>_<computeUnits>_<modelName>.json` を
吸い出した先（典型は `audio/<archive>/debug/compute_plan/`）。glob 可。

出力:
    --md で指定したファイル（既定: `docs/2026-05-19/mlcomputeplan-dispatch-map.md`）

使い方:
    scripts/aggregate_compute_plan.py audio/iPhone_*_phase4_*/debug/compute_plan/*.json

各 JSON は ComputePlanInspector が書いたフォーマット:
    {
      "precision": "float32",
      "computeUnits": "cpuAndGPU",
      "modelName": "Transformer_Decoder_fp32_fixed262",
      "operationCount": N,
      "dispatchSummary": {"cpu": x, "gpu": y, "neuralEngine": z, "unknown": w},
      "operations": [
        {"operatorName": "...", "preferredDevice": "cpu", "supportedDevices": [...], ...},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_plans(patterns: list[str]) -> list[dict]:
    plans: list[dict] = []
    for pat in patterns:
        for path_str in glob.glob(pat, recursive=True):
            path = Path(path_str)
            if not path.is_file() or path.suffix != ".json":
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                print(f"# warn: failed to parse {path}: {e}", file=sys.stderr)
                continue
            data["_source"] = str(path)
            plans.append(data)
    return plans


def format_summary_table(plans: list[dict]) -> list[str]:
    lines = []
    lines.append("## 1. ディスパッチサマリー（model × precision × computeUnits）")
    lines.append("")
    lines.append("| model | precision | computeUnits | ops | cpu | gpu | NE | unknown |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    plans_sorted = sorted(
        plans,
        key=lambda p: (p.get("modelName", ""), p.get("precision", ""), p.get("computeUnits", ""))
    )
    for p in plans_sorted:
        summary = p.get("dispatchSummary", {})
        lines.append(
            f"| {p.get('modelName','?')} | {p.get('precision','?')} | "
            f"{p.get('computeUnits','?')} | {p.get('operationCount', 0)} | "
            f"{summary.get('cpu',0)} | {summary.get('gpu',0)} | "
            f"{summary.get('neuralEngine',0)} | {summary.get('unknown',0)} |"
        )
    lines.append("")
    return lines


def format_op_distribution(plans: list[dict]) -> list[str]:
    """各 plan の op kind 別 dispatch 分布。"""
    lines = []
    lines.append("## 2. operator 種別 × device の分布")
    lines.append("")
    lines.append("各 (model, precision, computeUnits) について、operator 種別ごとに")
    lines.append("どのデバイスに行ったか。GPU 行きの最初の op を Phase 3 で重点的に見る。")
    lines.append("")
    for p in sorted(plans, key=lambda x: (x.get("modelName",""), x.get("precision",""), x.get("computeUnits",""))):
        header = f"### {p.get('modelName','?')} / {p.get('precision','?')} / {p.get('computeUnits','?')}"
        lines.append(header)
        lines.append("")
        ops = p.get("operations", [])
        kind_dev: dict[str, Counter] = defaultdict(Counter)
        for op in ops:
            kind = op.get("operatorName", "?")
            dev = op.get("preferredDevice", "unknown")
            kind_dev[kind][dev] += 1
        if not kind_dev:
            lines.append("（operations なし。NeuralNetwork 形式かも）")
            lines.append("")
            continue
        lines.append("| operator | cpu | gpu | NE | unknown | total |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for kind in sorted(kind_dev.keys(), key=lambda k: -sum(kind_dev[k].values())):
            counts = kind_dev[kind]
            total = sum(counts.values())
            lines.append(
                f"| {kind} | {counts.get('cpu',0)} | {counts.get('gpu',0)} | "
                f"{counts.get('neuralEngine',0)} | {counts.get('unknown',0)} | {total} |"
            )
        lines.append("")
    return lines


def format_first_gpu_ops(plans: list[dict]) -> list[str]:
    """各 plan で「GPU 行きの最初の op」を抽出して並べる。"""
    lines = []
    lines.append("## 3. 各 plan の GPU 行き最初の op（Phase 3 観測候補）")
    lines.append("")
    lines.append("| model | precision | computeUnits | first GPU op index | operatorName | outputName | weight |")
    lines.append("|---|---|---|---:|---|---|---:|")
    for p in sorted(plans, key=lambda x: (x.get("modelName",""), x.get("precision",""), x.get("computeUnits",""))):
        ops = p.get("operations", [])
        first_gpu = None
        for i, op in enumerate(ops):
            if op.get("preferredDevice") == "gpu":
                first_gpu = (i, op)
                break
        if first_gpu is None:
            lines.append(
                f"| {p.get('modelName','?')} | {p.get('precision','?')} | "
                f"{p.get('computeUnits','?')} | - | (none) | - | - |"
            )
        else:
            i, op = first_gpu
            outputs = op.get("outputNames", [])
            out_name = outputs[0] if outputs else "-"
            lines.append(
                f"| {p.get('modelName','?')} | {p.get('precision','?')} | "
                f"{p.get('computeUnits','?')} | {i} | {op.get('operatorName','?')} | "
                f"{out_name} | {op.get('estimatedCostWeight', 0):.4f} |"
            )
    lines.append("")
    return lines


def format_cross_diff(plans: list[dict]) -> list[str]:
    """F32×cpuAndGPU と F32×cpuOnly の Decoder で device が違う op を抜き出す。"""
    lines = []
    lines.append("## 4. F32 Decoder の cpuOnly vs cpuAndGPU で配置が変わる op")
    lines.append("")

    # precision は ModelPrecision.rawValue ("Float32") で書かれるので大文字小文字を無視して比較する
    decoders = [p for p in plans if "Decoder" in p.get("modelName","") and p.get("precision", "").lower() == "float32"]
    cpu_only = next((p for p in decoders if p.get("computeUnits") == "cpuOnly"), None)
    cpu_gpu = next((p for p in decoders if p.get("computeUnits") == "cpuAndGPU"), None)
    if cpu_only is None or cpu_gpu is None:
        lines.append("（F32 Decoder の cpuOnly と cpuAndGPU が両方そろっていない）")
        lines.append("")
        return lines

    # op の対応付けは output 名 + index で行う
    def key(op: dict, idx: int) -> str:
        outs = op.get("outputNames", [])
        return f"{idx}:{op.get('operatorName','?')}:{outs[0] if outs else ''}"

    only_map = {key(op, i): op for i, op in enumerate(cpu_only.get("operations", []))}
    gpu_map = {key(op, i): op for i, op in enumerate(cpu_gpu.get("operations", []))}
    diffs = []
    for k, op_o in only_map.items():
        op_g = gpu_map.get(k)
        if op_g is None:
            continue
        if op_o.get("preferredDevice") != op_g.get("preferredDevice"):
            diffs.append((k, op_o.get("preferredDevice"), op_g.get("preferredDevice"), op_g))

    if not diffs:
        lines.append("（差分なし。cpuAndGPU でも CPU に落ちている可能性）")
        lines.append("")
        return lines

    lines.append("| key (idx:operator:output) | cpuOnly | cpuAndGPU | weight |")
    lines.append("|---|---|---|---:|")
    for k, do, dg, op_g in diffs:
        lines.append(f"| `{k}` | {do} | {dg} | {op_g.get('estimatedCostWeight', 0):.4f} |")
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="compute_plan/*.json のパス（glob 可）")
    parser.add_argument("--md", default="docs/2026-05-19/mlcomputeplan-dispatch-map.md")
    args = parser.parse_args()

    plans = load_plans(args.inputs)
    if not plans:
        print("# no plan JSONs found", file=sys.stderr)
        return 1

    out_lines: list[str] = []
    out_lines.append("# Phase 4 — MLComputePlan dispatch マップ集計")
    out_lines.append("")
    out_lines.append(f"集計対象: {len(plans)} plan files")
    out_lines.append("")
    out_lines.append("生成元スクリプト: `scripts/aggregate_compute_plan.py`")
    out_lines.append("")
    out_lines.extend(format_summary_table(plans))
    out_lines.extend(format_first_gpu_ops(plans))
    out_lines.extend(format_cross_diff(plans))
    out_lines.extend(format_op_distribution(plans))

    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(out_lines))
    print(f"# wrote {md_path} ({len(plans)} plans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
