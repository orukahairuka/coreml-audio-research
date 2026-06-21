#!/usr/bin/env python3
"""PronounSE iOS 実機評価 ── 3軸（速い・壊れない・使える音）の研究用図表を生成する。

入力 CSV: data/2026-06-21/figures/*.csv
出力     : figures/*.png, figures/*.svg

生成する図:
  1. feasibility_map      成立マップ（3軸 ○× マトリクス）
  2. load_time_initial    初回ロード時間 棒グラフ（速い軸）
  3. load_time_cached     キャッシュ後ロード時間 棒グラフ（速い軸・補足）
  4. mcd_ranking          MCD ランキング 棒グラフ（使える音軸）
  5. stability_matrix     安定性 3×4 マトリクス（壊れない軸）

すべての図で「同じ 4 セル（F16/Int8 × {cpuAndNE, all}）が落ちる」ことを赤で統一表示する。

再生成:
    PronounSE/venv/bin/python scripts/make_figures.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---- パス ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "2026-06-21", "figures")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- 日本語フォント（無ければ英語ラベルにフォールバック）----
JP_FONT_CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
USE_JP = False
for fp in JP_FONT_CANDIDATES:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        USE_JP = True
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.dpi"] = 150


def T(ja, en):
    """日本語フォントがあれば日本語、無ければ英語を返す。"""
    return ja if USE_JP else en


# ---- 色（成立候補=緑 / 不成立=赤 で全図統一）----
GREEN = "#2e7d32"
GREEN_FILL = "#a5d6a7"
RED = "#c62828"
RED_FILL = "#ef9a9a"
GRAY = "#9e9e9e"


def group_color(group, fill=True):
    if group == "ane":
        return RED_FILL if fill else RED
    return GREEN_FILL if fill else GREEN


def read_csv(name):
    with open(os.path.join(DATA, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/{stem}.png / .svg")


def cell_label(precision, unit, two_line=True):
    sep = "\n" if two_line else " × "
    return f"{precision}{sep}{unit}"


# ===================================================================
# 1. 成立マップ（3軸 ○× マトリクス）
# ===================================================================
def fig_feasibility_map():
    rows = read_csv("feasibility.csv")
    axes_cols = [T("①速い", "1. Fast"),
                 T("②壊れない", "2. Robust"),
                 T("③使える音", "3. Usable")]
    verdict_col = T("成立", "Verdict")
    cols = axes_cols + [verdict_col]

    n_rows = len(rows)
    n_cols = len(cols)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, n_cols + 2.6)   # 左に行ラベル領域
    ax.set_ylim(0, n_rows + 1)
    ax.axis("off")

    label_w = 2.6
    cell_h = 1.0
    y0 = n_rows  # ヘッダ行

    # ヘッダ
    ax.text(label_w / 2, y0 + 0.5,
            T("設定グループ", "Configuration group"),
            ha="center", va="center", fontsize=12, fontweight="bold")
    for j, c in enumerate(cols):
        ax.text(label_w + j + 0.5, y0 + 0.5, c, ha="center", va="center",
                fontsize=12, fontweight="bold")

    for i, row in enumerate(rows):
        y = n_rows - 1 - i
        ane = row["verdict"] == "不成立"
        ec = RED if ane else GREEN
        # 行ラベル
        ax.add_patch(Rectangle((0, y), label_w, cell_h, facecolor="white",
                               edgecolor="#cccccc"))
        ax.text(0.1, y + cell_h / 2, row["group_label"].replace("\\n", "\n"),
                ha="left", va="center", fontsize=10)
        # 3軸セル
        for j, key in enumerate(("fast", "robust", "quality")):
            ok = row[key] == "1"
            ax.add_patch(Rectangle((label_w + j, y), 1, cell_h,
                                   facecolor=GREEN_FILL if ok else RED_FILL,
                                   edgecolor="white", linewidth=2))
            ax.text(label_w + j + 0.5, y + cell_h / 2, "○" if ok else "×",
                    ha="center", va="center", fontsize=20,
                    color=GREEN if ok else RED, fontweight="bold")
        # 成立判定セル
        ax.add_patch(Rectangle((label_w + 3, y), 1, cell_h,
                               facecolor=GREEN_FILL if not ane else RED_FILL,
                               edgecolor="white", linewidth=2))
        ax.text(label_w + 3 + 0.5, y + cell_h / 2,
                row["verdict"], ha="center", va="center", fontsize=11,
                color=ec, fontweight="bold")

    ax.set_title(
        T("インタラクティブ成立マップ ── 3軸すべてで同じ ANE 4セルが不成立",
          "Feasibility map: the same 4 ANE cells fail on all 3 axes"),
        fontsize=14, fontweight="bold", pad=14)
    save(fig, "feasibility_map")


# ===================================================================
# 2 & 3. ロード時間 棒グラフ
# ===================================================================
def _load_rows():
    return read_csv("load_time.csv")


def fig_load_time_initial():
    rows = _load_rows()
    labels = [cell_label(r["precision"], r["compute_unit"]) for r in rows]
    vals = [float(r["initial_mean_s"]) for r in rows]
    colors = [group_color(r["group"], fill=False) for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(range(len(rows)), vals, color=colors, edgecolor="black",
                  linewidth=0.4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(T("初回ロード時間 [秒]", "Initial load time [s]"), fontsize=12)
    ax.set_title(
        T("①速い軸 ── 初回モデルロード時間（run1/run2 平均, 3モデル合計）",
          "Axis 1 (Fast): initial model load time (mean of run1/run2)"),
        fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for i, (b, r, v) in enumerate(zip(bars, rows, vals)):
        txt = f"{v:.1f}s"
        if r["note"].startswith("run1"):
            txt += T("\n(run1 crash)", "\n(run1 crash)")
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, txt,
                ha="center", va="bottom", fontsize=8)

    legend = [Rectangle((0, 0), 1, 1, color=GREEN),
              Rectangle((0, 0), 1, 1, color=RED)]
    ax.legend(legend, [T("成立候補（CPU/GPU 系, ~1秒）", "Candidate (CPU/GPU, ~1s)"),
                       T("不成立（HiFi-GAN が ANE, 数十秒/crash）",
                         "Fail (HiFi-GAN on ANE, tens of s/crash)")],
              fontsize=10, loc="upper left")
    save(fig, "load_time_initial")


def fig_load_time_cached():
    rows = _load_rows()
    labels = [cell_label(r["precision"], r["compute_unit"]) for r in rows]
    vals = [float(r["cached_ms"]) for r in rows]
    colors = [group_color(r["group"], fill=False) for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.0))
    bars = ax.bar(range(len(rows)), vals, color=colors, edgecolor="black",
                  linewidth=0.4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(T("キャッシュ後ロード時間 [ミリ秒]", "Cached load time [ms]"),
                  fontsize=12)
    ax.set_title(
        T("①速い軸（補足）── キャッシュ後は全セル ~0.1〜0.2秒に収束",
          "Axis 1 (Fast, note): after caching all cells converge to ~0.1-0.2s"),
        fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    legend = [Rectangle((0, 0), 1, 1, color=GREEN),
              Rectangle((0, 0), 1, 1, color=RED)]
    ax.legend(legend, [T("成立候補（CPU/GPU 系）", "Candidate (CPU/GPU)"),
                       T("不成立 ANE 系", "Fail (ANE)")],
              fontsize=10, loc="upper left")
    save(fig, "load_time_cached")


# ===================================================================
# 4. MCD ランキング
# ===================================================================
def fig_mcd_ranking():
    rows = read_csv("mcd.csv")  # rank 昇順で格納済み
    labels = [cell_label(r["precision"], r["compute_unit"]) for r in rows]
    vals = [float(r["mcd_db"]) for r in rows]
    colors = [group_color(r["group"], fill=False) for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(range(len(rows)), vals, color=colors, edgecolor="black",
                  linewidth=0.4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(T("MCD [dB]（対 PyTorch baseline・小さいほど良い）",
                    "MCD [dB] (vs PyTorch baseline, lower=better)"), fontsize=11)
    ax.set_title(
        T("③使える音軸 ── MCD ランキング（昇順）。8位と9位の間に約5dBのギャップ",
          "Axis 3 (Usable): MCD ranking. ~5 dB gap between rank 8 and 9"),
        fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)

    # 8位と9位の境界に区切り線
    ax.axvline(7.5, color="black", linestyle="--", linewidth=1.5)
    ymid = (vals[7] + vals[8]) / 2
    ax.annotate(T("約5dB ギャップ\n（ここで明確に分離）",
                  "~5 dB gap\n(clear separation)"),
                xy=(7.5, ymid), xytext=(5.4, vals[8] + 0.6),
                fontsize=11, fontweight="bold", color="black",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="black"))

    legend = [Rectangle((0, 0), 1, 1, color=GREEN),
              Rectangle((0, 0), 1, 1, color=RED)]
    ax.legend(legend, [T("成立候補（CPU/GPU 系, 3.9〜6.6dB）",
                         "Candidate (CPU/GPU, 3.9-6.6 dB)"),
                       T("不成立（ANE 系, 11.6〜12.0dB）",
                         "Fail (ANE, 11.6-12.0 dB)")],
              fontsize=10, loc="upper left")
    save(fig, "mcd_ranking")


# ===================================================================
# 5. 安定性 3×4 マトリクス
# ===================================================================
def fig_stability_matrix():
    rows = read_csv("stability.csv")
    precisions = ["Float32", "Float16", "Int8"]
    units = ["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"]
    status = {(r["precision"], r["compute_unit"]): r["status"] for r in rows}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, len(units) + 1)
    ax.set_ylim(0, len(precisions) + 1)
    ax.axis("off")

    # 列ヘッダ
    for j, u in enumerate(units):
        ax.text(1 + j + 0.5, len(precisions) + 0.5, u, ha="center",
                va="center", fontsize=12, fontweight="bold")
    # 行ヘッダ + セル
    for i, p in enumerate(precisions):
        y = len(precisions) - 1 - i
        ax.text(0.5, y + 0.5, p, ha="center", va="center", fontsize=12,
                fontweight="bold")
        for j, u in enumerate(units):
            st = status.get((p, u), "")
            clipped = st == "clipped"
            ax.add_patch(Rectangle((1 + j, y), 1, 1,
                                   facecolor=RED_FILL if clipped else GREEN_FILL,
                                   edgecolor="white", linewidth=3))
            mark = "×" if clipped else "○"
            label = T("clipped", "clipped") if clipped else T("normal", "normal")
            col = RED if clipped else GREEN
            ax.text(1 + j + 0.5, y + 0.62, mark, ha="center", va="center",
                    fontsize=18, color=col, fontweight="bold")
            ax.text(1 + j + 0.5, y + 0.28, label, ha="center", va="center",
                    fontsize=9, color=col)

    ax.set_title(
        T("②壊れない軸 ── 安定性マトリクス（HiFi-GAN が ANE 実行の4セルで clipping）",
          "Axis 2 (Robust): stability matrix (clipping on 4 ANE cells)"),
        fontsize=13, fontweight="bold", pad=12)
    save(fig, "stability_matrix")


def main():
    print(f"japanese font: {'ON (' + plt.rcParams['font.family'][0] + ')' if USE_JP else 'OFF (english fallback)'}")
    fig_feasibility_map()
    fig_load_time_initial()
    fig_load_time_cached()
    fig_mcd_ranking()
    fig_stability_matrix()
    print("done.")


if __name__ == "__main__":
    main()
