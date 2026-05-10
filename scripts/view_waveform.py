"""result/output_<P>_<U>.wav の振幅波形を 12 通りグリッドで可視化するスクリプト

使い方:
    # 全12通りを1枚のグリッドで表示
    PronounSE/venv/bin/python scripts/view_waveform.py

    # 全12通りを PNG に保存 (figures/ 配下推奨)
    PronounSE/venv/bin/python scripts/view_waveform.py --save figures/waveform_grid.png

    # 特定の1ファイルだけ詳しく見る
    PronounSE/venv/bin/python scripts/view_waveform.py output_Float16_cpuAndGPU.wav
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV_DIR = os.path.join(REPO_ROOT, "result")
INPUT_WAV = os.path.join(
    REPO_ROOT, "ios", "CoreMLAudioApp", "CoreMLAudioApp", "Input", "input_sample.wav"
)

PRECISIONS = ["Float32", "Float16", "Int8"]
DEVICES = ["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"]


def _finalize(save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"保存しました: {save_path}")
    else:
        plt.show()


def _load_wav(path):
    rate, data = wavfile.read(path)
    # ステレオなら平均してモノラルに
    if data.ndim == 2:
        data = data.mean(axis=1)
    # int16 なら正規化して見やすく
    if np.issubdtype(data.dtype, np.integer):
        max_int = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_int
    return rate, data


def show_one(filename: str, save_path=None) -> None:
    path = os.path.join(WAV_DIR, filename)
    if not os.path.isfile(path):
        print(f"ファイルが見つかりません: {path}")
        sys.exit(1)

    rate, data = _load_wav(path)
    duration = len(data) / rate
    t = np.arange(len(data)) / rate
    print(f"file    : {filename}")
    print(f"rate    : {rate} Hz")
    print(f"samples : {len(data)} ({duration:.2f} s)")
    print(f"値域    : {data.min():.3f} 〜 {data.max():.3f}")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, data, linewidth=0.5)
    ax.set_title(filename)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _finalize(save_path)


def show_grid(save_path=None) -> None:
    # 1 行目: 入力波形 (最初の列のみ表示、残りは非表示)、
    # 2 行目以降: 出力波形 (3 精度 × 4 デバイス)
    fig, axes = plt.subplots(
        len(PRECISIONS) + 1,
        len(DEVICES),
        figsize=(16, 9),
        sharex=False,  # 入力と出力で長さが違いうるので X 軸は揃えない
        sharey=True,
    )

    # 1 行目: 入力波形
    if os.path.isfile(INPUT_WAV):
        rate, data = _load_wav(INPUT_WAV)
        t = np.arange(len(data)) / rate
        axes[0, 0].plot(t, data, linewidth=0.4)
        axes[0, 0].set_title("input", fontsize=10)
        axes[0, 0].set_ylabel("amplitude")
        axes[0, 0].grid(True, alpha=0.3)
        for j in range(1, len(DEVICES)):
            axes[0, j].axis("off")
    else:
        for j in range(len(DEVICES)):
            axes[0, j].axis("off")

    # 2 行目以降: 出力波形
    for i, prec in enumerate(PRECISIONS, start=1):
        for j, dev in enumerate(DEVICES):
            ax = axes[i, j]
            path = os.path.join(WAV_DIR, f"output_{prec}_{dev}.wav")
            if not os.path.isfile(path):
                ax.axis("off")
                continue
            rate, data = _load_wav(path)
            t = np.arange(len(data)) / rate
            ax.plot(t, data, linewidth=0.4)
            ax.set_title(f"{prec} / {dev}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == len(PRECISIONS):
                ax.set_xlabel("time (s)")
            if j == 0:
                ax.set_ylabel("amplitude")

    plt.tight_layout()
    _finalize(save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="result/ の wav 波形を可視化")
    parser.add_argument(
        "filename",
        nargs="?",
        help="特定の .wav ファイルだけ表示 (省略時は 12 通りグリッド)",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="表示せず PNG に保存する出力パス",
    )
    args = parser.parse_args()

    if args.filename:
        show_one(args.filename, save_path=args.save)
    else:
        if args.save:
            print(f"全12通り波形をグリッドで {args.save} に保存します...")
        else:
            print("全12通り波形を並べて表示します (matplotlib ウィンドウ)...")
        show_grid(save_path=args.save)


if __name__ == "__main__":
    main()
