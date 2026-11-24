"""可変長の実機テスト（VariableLengthTests）の出力を、PyTorch 版と比べる

実機の `Documents/Result/variable_length/` を取り出したフォルダ（--device）と、
`scripts/verify_variable_length.py` の出力フォルダ（--reference。PyTorch の結果と、実機に入れたメルがある）を渡す。
同じメル（PyTorch の前処理）を入れた経路どうしを比べる。iOS の前処理を通した経路は長さと NaN/Inf だけを見る
（前処理が違うので値は比べない）。

使い方
  PronounSE/venv/bin/python scripts/compare_variable_length_device.py \
      --device metrics/variable_length_<日付>/device_<機種>_f32_cpuOnly \
      --reference metrics/variable_length_<日付>/mac_range1_f32_cpu
"""

import argparse
import csv
import json
import os

import numpy as np


def diff(a, b):
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    e = np.abs(a64 - b64)
    snr = 10 * np.log10(np.sum(a64 ** 2) / max(np.sum((a64 - b64) ** 2), 1e-300))
    return {"max_abs": float(e.max()), "snr_db": float(snr),
            "cosine": float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-suffix", default="",
                    help="reference 側のファイル名で、名前の後ろに付く文字列（例: _22k）")
    args = ap.parse_args()

    dev = json.load(open(os.path.join(args.device, "summary.json")))
    rows = []
    for r in dev["rows"]:
        n = r["name"]
        rn = n + args.reference_suffix
        pt_post = np.load(os.path.join(args.reference, f"{rn}_pytorch_postnet.npy"))
        pt_wave = np.load(os.path.join(args.reference, f"{rn}_pytorch_wave.npy"))
        pt_mel = np.load(os.path.join(args.reference, f"{rn}_mel_in.npy"))
        ios_post = np.load(os.path.join(args.device, f"{n}_ios_postnet.npy"))
        ios_wave = np.load(os.path.join(args.device, f"{n}_ios_wave.npy")).ravel()
        same_len = ios_post.shape == pt_post.shape and ios_wave.size == pt_wave.size
        row = {
            "name": n, "T": r["T"], "memory_T": r["memory_T"], "postnet_T": r["postnet_T"],
            "samples": r["samples"], "expected_samples": r["expected_samples"], "same_length_as_pytorch": same_len,
            "nonfinite": r["nonfinite"], "ios_peak": float(np.abs(ios_wave).max()),
            "pytorch_peak": float(np.abs(pt_wave).max()),
            "ios_clipped": int((np.abs(ios_wave) >= 1).sum()), "pytorch_clipped": int((np.abs(pt_wave) >= 1).sum()),
            "postnet": diff(pt_post, ios_post) if same_len else None,
            "wave": diff(pt_wave, ios_wave) if same_len else None,
            "iosfront_T": r["iosfront_T"], "iosfront_samples": r["iosfront_samples"],
            "iosfront_nonfinite": r["iosfront_nonfinite"], "device_seconds": r["seconds"],
        }
        # iOS の前処理を通した経路（メルと最終波形）。前処理の条件が PyTorch 側と同じときだけ値を比べる意味がある
        front_mel_path = os.path.join(args.device, f"{n}_iosfront_mel.npy")
        if os.path.exists(front_mel_path):
            front_mel = np.load(front_mel_path)
            front_wave = np.load(os.path.join(args.device, f"{n}_iosfront_wave.npy")).ravel()
            row["iosfront_mel"] = diff(pt_mel, front_mel) if front_mel.shape == pt_mel.shape else None
            row["iosfront_wave"] = diff(pt_wave, front_wave) if front_wave.size == pt_wave.size else None
        rows.append(row)
        print(f"{n}: T={row['T']} 切り詰めなし={row['memory_T'] == row['postnet_T'] == row['T']} "
              f"長さ={row['samples']}/{row['expected_samples']} NaN/Inf={row['nonfinite']} "
              f"peak iOS {row['ios_peak']:.3f} / PyTorch {row['pytorch_peak']:.3f} "
              f"clip {row['ios_clipped']} / {row['pytorch_clipped']} "
              f"postnet SNR {row['postnet']['snr_db']:.1f} dB 波形 SNR {row['wave']['snr_db']:.1f} dB "
              f"(cos {row['wave']['cosine']:.6f}) | iOS 前処理 T={row['iosfront_T']} 長さ={row['iosfront_samples']} "
              f"| 実機 {row['device_seconds']:.1f} 秒")
        if row.get("iosfront_mel"):
            print(f"    iOS 前処理: メル max|差| {row['iosfront_mel']['max_abs']:.2e}  "
                  f"最終波形 SNR {row['iosfront_wave']['snr_db']:.1f} dB (cos {row['iosfront_wave']['cosine']:.6f})")

    with open(os.path.join(args.device, "comparison.json"), "w") as f:
        json.dump({"device_meta": {k: v for k, v in dev.items() if k != "rows"}, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    with open(os.path.join(args.device, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "T", "samples", "expected_samples", "nonfinite", "ios_peak", "pytorch_peak", "ios_clipped",
                    "pytorch_clipped", "postnet_snr_db", "wave_snr_db", "wave_cosine", "iosfront_T", "device_seconds"])
        for r in rows:
            w.writerow([r["name"], r["T"], r["samples"], r["expected_samples"], r["nonfinite"], f"{r['ios_peak']:.4f}",
                        f"{r['pytorch_peak']:.4f}", r["ios_clipped"], r["pytorch_clipped"],
                        f"{r['postnet']['snr_db']:.1f}", f"{r['wave']['snr_db']:.1f}", f"{r['wave']['cosine']:.6f}",
                        r["iosfront_T"], f"{r['device_seconds']:.2f}"])


if __name__ == "__main__":
    main()
