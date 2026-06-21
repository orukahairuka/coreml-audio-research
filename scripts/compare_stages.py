#!/usr/bin/env python3
"""ステージ境界の中間出力を、リファレンス dir vs 対象 run dir で段階横断比較する。

reference / run のどちらも「同じファイル名の .npy が並んだ dir」を取る。
  - reference に PyTorch baseline（extract_pytorch_stages.py の出力）を置けば
    「PyTorch vs CoreML(実機)」の絶対比較になる。
    ※ ただし iOS の mel フロントエンドは librosa と一致しない（フレーム数・正規化が違う）ため、
      これは純粋な ANE 切り分けではなく「絶対アンカー」。
  - reference に実機の良 run（例: f16 cpuAndGPU）を置けば、全 run が同一 mel_normalized
    （sha256 一致）を共有するので、差を compute unit だけに帰属できる混入なしの切り分けになる。

各ステージについて以下を出す:
    shape / dtype / min / max / mean / std / MAE / RMSE / cosine / peak / RMS / NaN・Inf / 判定

判定基準は stage_metrics.judge を参照（cosine と rms_ratio のヒューリスティック）。

実行例:
    # 実機 良 vs 壊れ（混入なしの切り分け）
    PronounSE/venv/bin/python scripts/compare_stages.py \\
        --reference audio/.../debug/<good f16 cpuAndGPU run> \\
        --run audio/.../debug/<f16 cpuAndNE run> \\
        --run audio/.../debug/<int8 cpuAndNE run> \\
        --md docs/2026-06-21/stage-compare-device.md --plots

    # PyTorch vs 実機（絶対アンカー）
    PronounSE/venv/bin/python scripts/compare_stages.py \\
        --reference data/2026-06-21/pytorch_reference/input_sample \\
        --run audio/.../debug/<run> --md docs/2026-06-21/stage-compare-pytorch.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_metrics as sm  # noqa: E402

# (ファイル名, 表示名, kind) — パイプライン順
STAGES = [
    ("mel_normalized.npy", "mel(encoder入力)", "tensor"),
    ("encoder_output.npy", "encoder出力", "tensor"),
    ("postnet_output.npy", "postnet(HiFi-GAN入力)", "tensor"),
    ("waveform_predeemph.npy", "HiFi-GAN出力波形", "waveform"),
    ("waveform_postdeemph.npy", "最終波形", "waveform"),
]


def run_label(path: Path) -> str:
    return path.name


def save_plot(ref: np.ndarray, cur: np.ndarray, stage_file: str, label: str, plot_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"# matplotlib 無効のため plot skip: {e}", file=sys.stderr)
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    base = stage_file.replace(".npy", "")
    a = np.asarray(ref).squeeze()
    b = np.asarray(cur).squeeze()

    if a.ndim == 2 and b.ndim == 2:
        n = min(a.shape[0], b.shape[0])
        diff = np.abs(a[:n].astype(np.float64) - b[:n].astype(np.float64))
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(diff.T, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"|ref - {label}|  {base}")
        ax.set_xlabel("frame")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{base}__{label}__heatmap.png", dpi=140)
        plt.close(fig)
    else:
        a1 = a.reshape(-1).astype(np.float64)
        b1 = b.reshape(-1).astype(np.float64)
        n = min(a1.size, b1.size)
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        axes[0].plot(a1[:n], lw=0.4, label="ref")
        axes[0].plot(b1[:n], lw=0.4, alpha=0.7, label=label)
        axes[0].axhline(1.0, color="r", lw=0.5, ls="--")
        axes[0].axhline(-1.0, color="r", lw=0.5, ls="--")
        axes[0].set_title(f"{base}  (ref vs {label}, 赤線=±1.0 clip)")
        axes[0].legend(loc="upper right")
        axes[1].plot(np.abs(a1[:n] - b1[:n]), lw=0.4, color="k")
        axes[1].set_title("abs diff")
        axes[1].set_xlabel("sample")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{base}__{label}__wave.png", dpi=140)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="リファレンス dir")
    parser.add_argument("--run", action="append", required=True, help="比較対象 run dir（複数可）")
    parser.add_argument("--md", help="Markdown 出力先")
    parser.add_argument("--plots", action="store_true", help="差分 plot を保存する")
    parser.add_argument("--plot-dir", help="plot 保存先（既定: --md と同じ dir の plots/）")
    args = parser.parse_args()

    ref_dir = Path(args.reference)
    if not ref_dir.is_dir():
        print(f"# reference not found: {ref_dir}", file=sys.stderr)
        return 1

    plot_dir = None
    if args.plots:
        if args.plot_dir:
            plot_dir = Path(args.plot_dir)
        elif args.md:
            plot_dir = Path(args.md).parent / "plots"
        else:
            plot_dir = ref_dir / "plots"

    lines: list[str] = []
    lines.append(f"reference: `{ref_dir}`")
    lines.append("")
    lines.extend(sm.md_table_header())

    print(f"# reference = {ref_dir}")
    print("  ".join(["stage", "condition", "cosine", "rms_ratio", "判定"]))

    for run_str in args.run:
        run_dir = Path(run_str)
        if not run_dir.is_dir():
            print(f"# run not found, skip: {run_dir}", file=sys.stderr)
            continue
        label = run_label(run_dir)
        for stage_file, stage_name, kind in STAGES:
            ref_path = ref_dir / stage_file
            run_path = run_dir / stage_file
            if not ref_path.exists() or not run_path.exists():
                continue
            ref_arr = np.load(ref_path)
            cur_arr = np.load(run_path)
            m = sm.compare(ref_arr, cur_arr, kind=kind)
            lines.append(sm.md_row(stage_name, label, m))
            print(f"  {stage_name:22s} {label:30s} cos={m['cosine']:.4f} "
                  f"rms_ratio={m['rms_ratio']:.3f} {m['verdict']}")
            if plot_dir is not None:
                save_plot(ref_arr, cur_arr, stage_file, label, plot_dir)
        lines.append("")  # run ごとに空行

    md_text = "\n".join(lines) + "\n"
    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md_text)
        print(f"# wrote {out}")
        if plot_dir is not None:
            print(f"# plots -> {plot_dir}")
    else:
        print("\n" + md_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
