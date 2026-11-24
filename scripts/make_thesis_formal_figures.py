#!/usr/bin/env python3
"""卒論第4章の正式評価の図を、集計済みの CSV から作る（測定データは変更しない）。

  figure4-rtf.png       入力のフレーム数と RTF（RTF = 1 と H200 の平均を線で示す）
  figure4-fad-speaker.png  話者ごとと Whole の FAD（通常版と構成 A〜C）
  figure4-fad-diff.png  Whole の FAD の構成間の差と 95% 区間（参照音を単位とするブートストラップ法）

実行例（PronounSE の venv）:
    PronounSE/venv/bin/python scripts/make_thesis_formal_figures.py --out-dir <出力先>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RTF_CSV = REPO / "metrics/rtf_official/rtf_official_20261123/summary/per_input_rtf.csv"
FAD_DIR = REPO / "metrics/fad_official_fadtk/bootstrap_20261123"
H200_RTF_MEAN = 0.317  # 開発者提供の値（summary/h200_comparison.csv）

# 構成ごとの色と形。色だけに頼らないよう形も変える（白黒印刷でも区別できる）
STYLE = {
    "original": dict(label="通常版", color="#6f6f6a", marker="o", mfc="none"),
    "A": dict(label="A：FP32", color="#2a78d6", marker="s", mfc="#2a78d6"),
    "B": dict(label="B：FP16", color="#eb6834", marker="^", mfc="#eb6834"),
    "C": dict(label="C：INT8", color="#1baf7a", marker="D", mfc="none"),
}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

plt.rcParams.update({
    "font.family": "Hiragino Sans", "axes.unicode_minus": False, "savefig.dpi": 200,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def figure_rtf(out: Path) -> None:
    rows = read(RTF_CSV)
    T = [int(r["T"]) for r in rows]
    fig, (ax_a, ax_bc) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    panels = [(ax_a, ["A"], "A：FP32"), (ax_bc, ["B", "C"], "B：FP16、C：INT8")]
    for ax, keys, title in panels:
        for k in keys:
            s = STYLE[k]
            ax.plot(T, [float(r[f"{k}_rtf_mean"]) for r in rows], linestyle="none", marker=s["marker"], ms=4,
                    color=s["color"], mfc=s["mfc"], mew=0.9, label=s["label"])
        ax.axhline(1.0, color=INK, lw=1.0, ls="--")
        ax.text(700, 1.0, "RTF = 1", fontsize=8, color=INK, va="bottom", ha="right")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("入力のフレーム数 T", fontsize=9)
        ax.set_xlim(0, 750)
        ax.set_ylim(bottom=0)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(labelsize=8)
    ax_bc.axhline(H200_RTF_MEAN, color=MUTED, lw=1.0, ls=":")
    ax_bc.text(745, H200_RTF_MEAN, "H200（開発者）の平均 0.317", fontsize=8, color=MUTED, va="bottom", ha="right")
    ax_bc.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    ax_a.set_ylabel("RTF（10回の平均）", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figure4-rtf.png")
    plt.close(fig)


def figure_fad_speaker(out: Path) -> None:
    rows = read(FAD_DIR / "fad_bootstrap_by_config.csv")
    fad = {(r["config"], r["condition"]): float(r["fad"]) for r in rows}
    conds = ["m-01", "m-02", "m-03", "m-04", "m-05", "f-01", "Whole"]
    keys = [("original", "original"), ("A", "fp32"), ("B", "fp16"), ("C", "int8")]
    offs = [-0.24, -0.08, 0.08, 0.24]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    for (k, cfg), dx in zip(keys, offs):
        s = STYLE[k]
        xs = [i + dx for i in range(len(conds))]
        ax.plot(xs, [fad[(cfg, c)] for c in conds], linestyle="none", marker=s["marker"], ms=6,
                color=s["color"], mfc=s["mfc"], mew=1.1, label=s["label"])
    ax.axvline(len(conds) - 1.5, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(["m-01", "m-02", "m-03", "m-04", "m-05", "f-01", "Whole\n（600件）"], fontsize=9)
    ax.set_xlabel("話者（各100件）", fontsize=9)
    ax.set_ylabel("参照音とのFAD", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.6)
    fig.tight_layout()
    fig.savefig(out / "figure4-fad-speaker.png")
    plt.close(fig)


def signed(text: str) -> str:
    """ハイフンをマイナス記号にする（本文の表記と合わせる）"""
    return text.replace("-", "−")


def figure_fad_diff(out: Path) -> None:
    rows = {r["comparison"]: r for r in read(FAD_DIR / "delta_fad_ci.csv") if r["condition"] == "Whole"}
    order = [("fp32 - original", "A − 通常版"), ("fp16 - original", "B − 通常版"), ("int8 - original", "C − 通常版"),
             ("fp16 - fp32", "B − A"), ("int8 - fp32", "C − A"), ("int8 - fp16", "C − B")]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    n = len(order)
    for i, (key, label) in enumerate(order):
        r = rows[key]
        y = n - 1 - i
        d, lo, hi = float(r["fad_delta"]), float(r["fad_ci95_low"]), float(r["fad_ci95_high"])
        excludes_zero = lo > 0 or hi < 0
        ax.plot([lo, hi], [y, y], color=INK, lw=1.6, solid_capstyle="butt")
        ax.plot([d], [y], marker="o", ms=7, color=INK, mfc=INK if excludes_zero else "white", mew=1.4)
        ax.text(1.03, y, signed(f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]"), fontsize=8, color=INK, va="center",
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color=MUTED, lw=1.0, ls="--")
    ax.axhline(2.5, color=GRID, lw=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for _, label in reversed(order)], fontsize=9)
    ax.set_xlim(-2.6, 1.4)
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_xticklabels(["−2", "−1", "0", "+1"])
    ax.set_xlabel("WholeのFADの差（負なら左の構成のほうが小さい）", fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.tick_params(labelsize=8)
    ax.text(1.03, n - 0.35, "差  [95%区間]", fontsize=8, color=MUTED, va="center", transform=ax.get_yaxis_transform())
    fig.tight_layout()
    fig.savefig(out / "figure4-fad-diff.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_rtf(args.out_dir)
    figure_fad_speaker(args.out_dir)
    figure_fad_diff(args.out_dir)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
