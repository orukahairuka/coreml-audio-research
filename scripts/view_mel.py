"""result/mel/ の .npy ファイルを読み込んで、matplotlib で可視化するスクリプト

使い方:
    # 全12通りを1枚のグリッドで表示
    PronounSE/venv/bin/python scripts/view_mel.py

    # 特定の1ファイルだけ詳しく見る
    PronounSE/venv/bin/python scripts/view_mel.py output_mel_Float16_cpuAndGPU.npy
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

MEL_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "result", "mel")


def show_one(filename: str) -> None:
    """1ファイルを詳しく表示する"""
    path = os.path.join(MEL_DIR, filename)
    if not os.path.isfile(path):
        print(f"ファイルが見つかりません: {path}")
        sys.exit(1)

    mel = np.load(path)
    print(f"file : {filename}")
    print(f"shape: {mel.shape}  (frames × n_mels)")
    print(f"dtype: {mel.dtype}")
    print(f"値域 : {mel.min():.2f} 〜 {mel.max():.2f} dB")
    print(f"平均 : {mel.mean():.2f} dB")

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mel.T, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
    ax.set_title(filename)
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    plt.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()
    plt.show()


def show_grid() -> None:
    """全12通り + 入力 を1枚のグリッドで表示"""
    precisions = ["Float32", "Float16", "Int8"]
    devices = ["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"]

    fig, axes = plt.subplots(
        len(precisions) + 1, len(devices),
        figsize=(16, 9),
        sharex=True, sharey=True,
    )

    # 1行目: 入力メル (1枚だけ、残りは非表示)
    input_mel = np.load(os.path.join(MEL_DIR, "input_mel.npy"))
    for j, _ in enumerate(devices):
        ax = axes[0, j]
        if j == 0:
            ax.imshow(input_mel.T, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
            ax.set_title("input_mel", fontsize=10)
            ax.set_ylabel("mel bin")
        else:
            ax.axis("off")

    # 2行目以降: 出力メル (3精度 × 4デバイス)
    for i, prec in enumerate(precisions, start=1):
        for j, dev in enumerate(devices):
            path = os.path.join(MEL_DIR, f"output_mel_{prec}_{dev}.npy")
            if not os.path.isfile(path):
                axes[i, j].axis("off")
                continue
            arr = np.load(path)
            ax = axes[i, j]
            ax.imshow(arr.T, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
            ax.set_title(f"{prec} / {dev}", fontsize=9)
            if i == len(precisions):
                ax.set_xlabel("frame")
            if j == 0:
                ax.set_ylabel("mel bin")

    plt.tight_layout()
    plt.show()


def main() -> None:
    if len(sys.argv) > 1:
        show_one(sys.argv[1])
    else:
        # 引数なし: グリッドで全部見せる
        files = sorted(f for f in os.listdir(MEL_DIR) if f.endswith(".npy"))
        print(f"{MEL_DIR} に {len(files)} 個の .npy ファイルがあります:")
        for f in files:
            print(f"  {f}")
        print()
        print("全12通り + 入力メル を並べて表示します (matplotlib ウィンドウ)...")
        show_grid()


if __name__ == "__main__":
    main()
