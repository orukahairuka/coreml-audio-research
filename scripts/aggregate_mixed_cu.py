#!/usr/bin/env python3
"""卒論第4章の条件 A〜D（HiFi-GAN のみ別の Compute Units を指定する比較）を集計する。

thesis_ch4_metrics.py の基準は Float32 + cpuOnly で、aggregate_thesis_ch4.py は 12 条件だけを扱うので、
条件 A〜D はこのスクリプトで集計する。

振幅特性の基準は、同じセッションの条件 D の、同じ入力のウォームアップ後の 1 回目の出力とする。
入力ごとにウォームアップ後の実行の中央値を求め、4 入力の範囲を示す。失敗した実行は件数として出す。

出力（--out-dir、既定は <計測ディレクトリ>/tables/）:
  mixed_cu_waveform.{csv,md}    振幅特性（RMS 比・コサイン類似度・範囲の超過率）
  mixed_cu_inference.{csv,md}   推論時間と推論 RTF
  mixed_cu_load.{csv,md}        モデルの読み込み時間

実行例:
  PronounSE/venv/bin/python scripts/aggregate_mixed_cu.py metrics/final_ch4v7_<日時>/iPhone_13/mixed_cu
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_ch4_metrics import compare, load_npy  # noqa: E402

PRECISION = "Float16"
CONDITIONS = {
    "A": "cpuAndNE",
    "B": "cpuAndNE+hifi-cpuAndGPU",
    "C": "cpuAndNE+hifi-cpuOnly",
    "D": "cpuOnly",
}
BASELINE = "D"
INPUTS = ["1", "6", "11", "2"]
LOAD_KINDS = ["after_install", "restart", "reload"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def median_of(rows: list[dict], column: str) -> float:
    values = [float(r[column]) for r in rows if r.get(column)]
    return st.median(values) if values else float("nan")


def select(rows: list[dict], label: str) -> list[dict]:
    return [r for r in rows if r["precision"] == PRECISION and r["compute_units"] == CONDITIONS[label]]


def steady_runs(rows: list[dict], name: str) -> list[dict]:
    selected = [r for r in rows if r["input_name"] == name and r["phase"] == "steady" and r["status"] == "success"]
    return sorted(selected, key=lambda r: int(r["run_index"]))


def write_table(out_dir: Path, name: str, header: list[str], rows: list[list], note: str) -> None:
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    (out_dir / f"{name}.md").write_text("\n".join(lines + ["", note]) + "\n")


def span(values: list[float], digits: int) -> str:
    return f"{min(values):.{digits}f}〜{max(values):.{digits}f}"


def table_waveform(out_dir: Path, infer: list[dict], device_dir: Path) -> None:
    baseline = {}
    for name in INPUTS:
        runs = steady_runs(select(infer, BASELINE), name)
        if not runs:
            raise SystemExit(f"基準（条件 {BASELINE}）の入力 {name} に成功した実行がありません")
        baseline[name] = load_npy(device_dir / f"{runs[0]['file_prefix']}_hifigan.npy")

    header = ["条件", "n", "失敗", "RMS比（4入力の範囲）", "コサイン類似度（4入力の範囲）", "範囲の超過率 [%]（4入力の範囲）",
              "NaN", "Inf"] + [f"入力{name}のRMS比" for name in INPUTS]
    rows = []
    for label in CONDITIONS:
        selected = select(infer, label)
        rms, cosine, over1, count = {}, [], [], 0
        for name in INPUTS:
            runs = steady_runs(selected, name)
            count += len(runs)
            results = [compare(baseline[name], load_npy(device_dir / f"{r['file_prefix']}_hifigan.npy")) for r in runs]
            rms[name] = st.median(m["rms_ratio"] for m in results)
            cosine.append(st.median(m["cosine"] for m in results))
            over1.append(median_of(runs, "deemph_over1_ratio") * 100)
        rows.append([
            label, count, sum(r["status"] != "success" for r in selected),
            span(list(rms.values()), 2), span(cosine, 2), span(over1, 1),
            sum(int(r["hifigan_nan"] or 0) for r in selected), sum(int(r["hifigan_inf"] or 0) for r in selected),
        ] + [f"{rms[name]:.3f}" for name in INPUTS])
    write_table(out_dir, "mixed_cu_waveform", header, rows,
                f"基準は条件 {BASELINE} の同じ入力のウォームアップ後の 1 回目。RMS 比とコサイン類似度は HiFi-GAN の出力、"
                "範囲の超過率はデエンファシス後の波形。")


def table_inference(out_dir: Path, infer: list[dict]) -> None:
    header = ["条件", "初回の推論 [ms]（入力1）", "ウォームアップ後 [ms]（入力1, n=5）", "推論RTF（4入力, n=20）", "推論RTFの最大",
              "Decoderの1回あたり [ms]", "HiFi-GAN [ms]", "合成処理時間 [ms]（入力1）", "ウォームアップ後の熱状態"]
    rows = []
    for label in CONDITIONS:
        selected = [r for r in select(infer, label) if r["status"] == "success"]
        initial = [r for r in selected if r["phase"] == "initial"]
        steady = [r for r in selected if r["phase"] == "steady"]
        first = [r for r in steady if r["input_name"] == INPUTS[0]]
        thermal = Counter(r["thermal_before"] for r in steady)
        rows.append([
            label, f"{median_of(initial, 'synth_ms'):,.0f}", f"{median_of(first, 'synth_ms'):,.0f}",
            f"{median_of(steady, 'rtf'):.2f}", f"{max(float(r['rtf']) for r in steady):.2f}",
            f"{median_of(steady, 'decoder_step_ms_median'):.1f}", f"{median_of(steady, 'hifigan_ms'):.0f}",
            f"{median_of(first, 'e2e_ms'):,.0f}", " ".join(f"{k}:{v}" for k, v in sorted(thermal.items())),
        ])
    write_table(out_dir, "mixed_cu_inference", header, rows, "中央値。推論時間はモデルの呼び出しの合計。")


def table_load(out_dir: Path, load: list[dict], host: list[dict]) -> None:
    kind_of = {r["launch_id"]: r.get("load_kind", "") for r in host}
    header = ["条件", "インストール直後の初回 [ms]", "n", "起動し直した後の初回 [ms]", "n", "再読み込み [ms]", "n"]
    rows = []
    for label in CONDITIONS:
        selected = [r for r in select(load, label) if r["status"] == "success"]
        groups = {kind: [] for kind in LOAD_KINDS}
        for r in selected:
            kind = kind_of.get(r["launch_id"], "") if r["kind"] == "first" else "reload"
            if kind in groups:
                groups[kind].append(r)
        row = [label]
        for kind in LOAD_KINDS:
            row += [f"{median_of(groups[kind], 'total_ms'):,.0f}", len(groups[kind])]
        rows.append(row)
    write_table(out_dir, "mixed_cu_load", header, rows, "3 つのモデルの読み込み時間の合計の中央値。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench_dir", type=Path, help="raw/ を含む条件 A〜D の計測ディレクトリ")
    parser.add_argument("--out-dir", type=Path, help="出力先（既定は <計測ディレクトリ>/tables）")
    args = parser.parse_args()

    raw = args.bench_dir / "raw"
    device_dir = raw / "device"
    out_dir = args.out_dir or args.bench_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    infer = read_csv(device_dir / "infer_runs.csv")
    table_waveform(out_dir, infer, device_dir)
    table_inference(out_dir, infer)
    table_load(out_dir, read_csv(device_dir / "load_runs.csv"), read_csv(raw / "host_runs.csv"))
    print(f"# tables → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
