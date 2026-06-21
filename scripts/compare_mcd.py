#!/usr/bin/env python3
"""baseline wav と複数の対象 wav の MCD（メルケプストラム歪み）を計算して表で出す。

MCD = 2音をフレームごとにメルケプストラム係数（音色ベクトル）へ変換し、
DTW で時刻を揃えて距離を取り、平均した値[dB]。小さいほど baseline に音色が近い。
「使える音」軸の足切り用（最終判断は聴感）。

使い方:
  # baseline と個別の wav を比較
  PronounSE/venv/bin/python scripts/compare_mcd.py --ref baseline.wav a.wav b.wav

  # ディレクトリ内の output_*.wav をまとめて比較
  PronounSE/venv/bin/python scripts/compare_mcd.py --ref baseline.wav --dir result/
"""
import argparse
import glob
import os

from pymcd.mcd import Calculate_MCD


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="基準(baseline)となる wav")
    ap.add_argument("--dir", help="比較対象 wav が入ったディレクトリ（output_*.wav を拾う）")
    ap.add_argument("targets", nargs="*", help="比較対象の wav（個別指定）")
    ap.add_argument("--mode", default="dtw", choices=["plain", "dtw", "dtw_sl"],
                    help="アライメント方式（既定 dtw: 長さズレを伸縮して揃える）")
    args = ap.parse_args()

    targets = list(args.targets)
    if args.dir:
        targets += sorted(glob.glob(os.path.join(args.dir, "output_*.wav")))
    if not targets:
        ap.error("比較対象がありません（--dir かファイルを指定してください）")

    mcd = Calculate_MCD(MCD_mode=args.mode)
    rows = []
    for t in targets:
        score = mcd.calculate_mcd(args.ref, t)
        rows.append((os.path.basename(t), score))
    rows.sort(key=lambda r: r[1])

    ref_name = os.path.basename(args.ref)
    print(f"\nMCD (mode={args.mode})  基準 = {ref_name}")
    print(f"{'対象':36s} {'MCD(dB)':>9s}")
    print("-" * 47)
    for name, score in rows:
        print(f"{name:36s} {score:9.3f}")
    print("\n※ 小さいほど baseline に近い。0 に近い＝ほぼ同じ音。")


if __name__ == "__main__":
    main()
