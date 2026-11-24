#!/usr/bin/env python3
"""卒論第4章と付録の図を、最終実装の計測結果（iPhone 13）から作る。

出力するファイルの名前は、論文の assets/image/ のファイルと同じにする。
  figure4-3.png  図4-1  HiFi-GAN の出力の RMS 比（合成全体と HiFi-GAN のみ）
  figure4-5.png  図4-2  合成処理時間の内訳
  figure4-1.png  付図1  3 つのモデルの読み込み時間の合計
  figure4-4.png  付図2  範囲の超過率
  figure4-2.png  付図3  入力1の推論 RTF

実行例（PronounSE の venv）:
    PronounSE/venv/bin/python scripts/make_thesis_final_figures.py \\
        metrics/final_ch4v7_<日時>/iPhone_13 --out-dir <出力先>
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRECISIONS = ["Float32", "Float16", "Int8"]
COMPUTE_UNITS = ["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"]
CONDITIONS = [(p, cu) for cu in COMPUTE_UNITS for p in PRECISIONS]
NE_ALLOWED = {(p, cu) for p in ("Float16", "Int8") for cu in ("cpuAndNE", "all")}
INPUTS = ["1", "6", "11", "2"]
INPUT_LABELS = ["入力1", "入力2", "入力3", "入力4"]
MARKERS = ["o", "s", "^", "D"]

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
AUDIO_SEC = 256 * 262 / 22050

plt.rcParams.update({
    "font.family": "Hiragino Sans", "axes.unicode_minus": False, "savefig.dpi": 200,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.titlecolor": TEXT2, "axes.spines.top": False, "axes.spines.right": False,
})


def rows(path: Path, success_only: bool = True) -> list[dict]:
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if not success_only or r.get("status", "success") == "success"]


def select(data: list[dict], condition: tuple[str, str], **fields) -> list[dict]:
    return [r for r in data if (r["precision"], r["compute_units"]) == condition
            and all(r[k] == v for k, v in fields.items())]


def median_of(data: list[dict], column: str) -> float:
    return st.median(float(r[column]) for r in data)


def group_axis(ax) -> None:
    """数値形式の目盛りの下に Compute Units を表示する"""
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([p for p, _ in CONDITIONS], fontsize=8)
    for index, unit in enumerate(COMPUTE_UNITS):
        ax.text(index * 3 + 1, -0.17, unit, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=10, fontweight="bold", color=TEXT2)
    for x in (2.5, 5.5, 8.5):
        ax.axvline(x, color=GRID, lw=1.2, zorder=0)
    ax.set_xlim(-0.7, len(CONDITIONS) - 0.3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def input_dots(ax, data: list[dict], column: str, scale: float = 1.0) -> None:
    ys = list(range(len(CONDITIONS)))
    for y, condition in zip(ys, CONDITIONS):
        if condition in NE_ALLOWED:
            ax.axhspan(y - 0.45, y + 0.45, color=ORANGE, alpha=0.10, lw=0, zorder=0)
    for k, (name, label, marker) in enumerate(zip(INPUTS, INPUT_LABELS, MARKERS)):
        xs = [median_of(select(data, c, input_name=name), column) * scale for c in CONDITIONS]
        ax.scatter(xs, [y + (k - 1.5) * 0.14 for y in ys], marker=marker, s=24, facecolor="none",
                   edgecolor=BLUE, linewidth=1.3, label=label, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{p} × {cu}" for p, cu in CONDITIONS], fontsize=8)
    for tick, condition in zip(ax.get_yticklabels(), CONDITIONS):
        if condition in NE_ALLOWED:
            tick.set_fontweight("bold")
            tick.set_color(TEXT)
    ax.set_ylim(len(CONDITIONS) - 0.5, -0.5)
    for y in (2.5, 5.5, 8.5):
        ax.axhline(y, color=GRID, lw=1, zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def figure_rms(out: Path, infer: list[dict], vocoder: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True, sharex=True)
    panels = [(infer, "合成全体（基準：Float32 × cpuOnly）"), (vocoder, "HiFi-GANのみ（基準：PyTorch版）")]
    for ax, (data, title) in zip(axes, panels):
        input_dots(ax, data, "hifigan_rms_ratio")
        ax.axvline(1.0, color=TEXT2, lw=1, ls="--", zorder=1)
        ax.set_xlim(0, 7.4)
        ax.set_xlabel("HiFi-GANの出力のRMS比", fontsize=9)
        ax.set_title(title, fontsize=9, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "figure4-3.png")
    plt.close(fig)


def figure_over(out: Path, infer: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    input_dots(ax, infer, "deemph_over1_ratio", scale=100)
    ax.set_xlim(left=-0.3)
    ax.set_xlabel("範囲の超過率 [%]（デエンファシス後）", fontsize=9)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "figure4-4.png")
    plt.close(fig)


def range_marks(ax, xs, groups, color, marker, label) -> None:
    medians = [st.median(g) for g in groups]
    lower = [m - min(g) for m, g in zip(medians, groups)]
    upper = [max(g) - m for m, g in zip(medians, groups)]
    ax.errorbar(xs, medians, yerr=[lower, upper], fmt=marker, ms=6, color=color, ecolor=color,
                elinewidth=1.5, capsize=3, label=label, zorder=3)


def figure_load(out: Path, load: list[dict], host: list[dict]) -> None:
    kind_of = {r["launch_id"]: r.get("load_kind", "") for r in host}
    series = [("after_install", "インストール直後の初回", BLUE, "o"), ("restart", "起動し直した後の初回", ORANGE, "s"),
              ("reload", "再読み込み", GREEN, "^")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k, (kind, label, color, marker) in enumerate(series):
        groups = []
        for condition in CONDITIONS:
            data = select(load, condition)
            picked = [r for r in data if (r["kind"] != "first" if kind == "reload"
                                          else r["kind"] == "first" and kind_of.get(r["launch_id"]) == kind)]
            groups.append([float(r["total_ms"]) for r in picked])
        range_marks(ax, [i + (k - 1) * 0.22 for i in range(len(CONDITIONS))], groups, color, marker, label)
    ax.set_yscale("log")
    ax.set_ylabel("3つのモデルの読み込み時間の合計 [ms]（対数）", fontsize=9)
    group_axis(ax)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out / "figure4-1.png")
    plt.close(fig)


def figure_rtf(out: Path, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    steady, initial = [], []
    for condition in CONDITIONS:
        data = select(runs, condition, input_name=INPUTS[0])
        steady.append([float(r["rtf"]) for r in data if r["phase"] == "steady"])
        initial.append([float(r["rtf"]) for r in data if r["phase"] == "initial"])
    range_marks(ax, [i - 0.13 for i in range(len(CONDITIONS))], steady, BLUE, "o", "ウォームアップ後（入力1、5回の中央値と最小〜最大）")
    range_marks(ax, [i + 0.13 for i in range(len(CONDITIONS))], initial, ORANGE, "D", "初回の推論（入力1、1回）")
    ax.axhline(1.0, color=TEXT2, lw=1, ls="--", zorder=1)
    ax.text(len(CONDITIONS) - 0.4, 1.03, "RTF = 1", ha="right", va="bottom", fontsize=8, color=TEXT2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("推論RTF（推論時間 ÷ 合成音の長さ）", fontsize=9)
    group_axis(ax)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out / "figure4-2.png")
    plt.close(fig)


def figure_breakdown(out: Path, runs: list[dict], stages: list[dict]) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.2), gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.55})
    steady = [r for r in runs if r["phase"] == "steady" and r["input_name"] == INPUTS[0]]
    model = [median_of(select(steady, c), "synth_ms") / 1000 for c in CONDITIONS]
    other = [(median_of(select(steady, c), "e2e_ms") - median_of(select(steady, c), "synth_ms")) / 1000 for c in CONDITIONS]
    xs = list(range(len(CONDITIONS)))
    top.bar(xs, model, 0.62, color=BLUE, edgecolor="white", linewidth=1, label="推論時間（モデルの呼び出し）")
    top.bar(xs, other, 0.62, bottom=model, color=ORANGE, edgecolor="white", linewidth=1, label="それ以外")
    top.axhline(AUDIO_SEC, color=TEXT2, lw=1.2, ls="--", label=f"合成音の長さ（{AUDIO_SEC:.2f} 秒）")
    top.set_ylabel("合成処理時間 [秒]（入力1、5回の中央値）", fontsize=9)
    top.set_title("12条件", fontsize=9, loc="left")
    group_axis(top)
    top.legend(frameon=False, fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    top.set_ylim(0, max(a + b for a, b in zip(model, other)) * 1.2)

    data = [r for r in stages if r["phase"] == "steady" and r["input_name"] == INPUTS[0]]
    parts = [("Decoderの呼び出し", "dec_predict_ms", BLUE), ("HiFi-GANの呼び出し", "voc_predict_ms", BLUE),
             ("Encoderの呼び出し", "enc_predict_ms", BLUE), ("モデルの呼び出し以外", "non_predict_ms", ORANGE)]
    values = [median_of(data, column) for _, column, _ in parts]
    ys = list(range(len(parts)))
    bottom.barh(ys, values, 0.6, color=[c for _, _, c in parts], edgecolor="white")
    for y, value in zip(ys, values):
        bottom.text(value + max(values) * 0.012, y, f"{value:,.1f} ms", va="center", fontsize=8, color=TEXT)
    bottom.set_yticks(ys)
    bottom.set_yticklabels([label for label, _, _ in parts], fontsize=8)
    bottom.invert_yaxis()
    bottom.set_xlim(0, max(values) * 1.14)
    bottom.set_xlabel("時間 [ms]（条件C、入力1、5回の中央値）", fontsize=9)
    bottom.set_title("条件Cの処理ごとの時間", fontsize=9, loc="left")
    bottom.grid(axis="x", color=GRID, lw=0.8)
    bottom.set_axisbelow(True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.08)
    fig.savefig(out / "figure4-5.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("device_dir", type=Path, help="metrics/final_ch4v7_<日時>/iPhone_13")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = args.device_dir / "conditions12/raw"
    infer = rows(raw / "infer_runs_metrics.csv")
    steady = [r for r in infer if r["phase"] == "steady"]
    vocoder = [r for r in rows(raw / "vocoder_runs_metrics.csv") if r["phase"] == "steady"]
    figure_rms(args.out_dir, steady, vocoder)
    figure_over(args.out_dir, steady)
    figure_load(args.out_dir, rows(raw / "device/load_runs.csv"), rows(raw / "host_runs.csv", success_only=False))
    figure_rtf(args.out_dir, infer)
    figure_breakdown(args.out_dir, infer, rows(args.device_dir / "e2e/profile/raw/device/infer_stages.csv"))
    print(f"# figures → {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
