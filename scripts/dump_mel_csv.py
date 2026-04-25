"""result/mel/ の .npy を CSV に変換する

使い方:
    # 全 .npy を CSV に変換 (result/mel/csv/ に出力)
    PronounSE/venv/bin/python scripts/dump_mel_csv.py

    # 特定の1ファイルだけ
    PronounSE/venv/bin/python scripts/dump_mel_csv.py output_mel_Float16_all.npy

CSV は Numbers.app や Excel で開けます。
"""

import os
import sys

import numpy as np

MEL_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "result", "mel")
OUT_DIR = os.path.join(MEL_DIR, "csv")


def dump_one(npy_name: str) -> str:
    src = os.path.join(MEL_DIR, npy_name)
    if not os.path.isfile(src):
        print(f"ファイルが見つかりません: {src}")
        sys.exit(1)

    arr = np.load(src)
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_name = npy_name.replace(".npy", ".csv")
    dst = os.path.join(OUT_DIR, csv_name)

    # ヘッダ行: mel_bin 0..255
    header = ",".join(f"mel{i}" for i in range(arr.shape[1]))
    np.savetxt(dst, arr, fmt="%.3f", delimiter=",", header=header, comments="")
    print(f"  {npy_name}: {arr.shape} → {dst}")
    return dst


def main() -> None:
    if len(sys.argv) > 1:
        out = dump_one(sys.argv[1])
        print(f"\n完了: {out}")
        print("Numbers.app で開く:  open", out)
    else:
        files = sorted(f for f in os.listdir(MEL_DIR) if f.endswith(".npy"))
        print(f"{len(files)} 個の .npy を CSV に変換します...")
        for f in files:
            dump_one(f)
        print(f"\n完了: {OUT_DIR}")
        print(f"Finder で開く:  open {OUT_DIR}")


if __name__ == "__main__":
    main()
