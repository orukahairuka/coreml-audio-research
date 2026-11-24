# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""卒論の図（入力音声の長さと推論時間）を正式 RTF 測定の集計から作る。

uv run scripts/rtf_official/make_thesis_figure.py <summary フォルダ> <出力 png>
入力は aggregate_rtf.py が出した per_input_time_s.csv だけ。測定データは変更しない。
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

summary, out = Path(sys.argv[1]), Path(sys.argv[2])
p = pd.read_csv(summary / "per_input_time_s.csv")

plt.rcParams.update({"font.family": "Hiragino Sans", "axes.unicode_minus": False, "savefig.dpi": 200})
panels = [("A", "A：FP32", "#2a78d6"), ("B", "B：FP16", "#eb6834"), ("C", "C：INT8（W8A16）", "#1baf7a")]
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), sharex=True)
for ax, (key, title, color) in zip(axes, panels):
    ax.errorbar(p["input_duration_s"], p[f"{key}_time_mean_s"], yerr=p[f"{key}_time_sd_s"], fmt="o", ms=3,
                color=color, ecolor="#555555", elinewidth=0.8, capsize=2)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("入力音声の長さ [s]", fontsize=9)
    ax.set_xlim(0, 9)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("推論時間 [s]（10回の平均）", fontsize=9)
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(out)
