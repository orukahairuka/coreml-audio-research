"""result/ 配下の timing JSON と mel .npy をまとめて 12 通りの指標表を作る

使い方:
    PronounSE/venv/bin/python scripts/aggregate_metrics.py

前提:
    - scripts/extract_ui_test_results.sh を実行済みで result/timing/ と result/mel/ が揃っていること
    - 各組み合わせにつき timing_<P>_<U>.json と output_mel_<P>_<U>.npy が対応している

出力:
    Precision × ComputeUnit 12 通りの表を標準出力に書き出す。
    速さ (totalMs / RTF / decoder per-step ms) ・サイズ (modelMB) ・音質 (mel L1/L2 vs Float32_cpuOnly) の3軸。
"""

import glob
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMING_DIR = os.path.join(REPO_ROOT, "result", "timing")
MEL_DIR = os.path.join(REPO_ROOT, "result", "mel")

# 比較基準: Float32 / cpuOnly を「真値」相当として扱う
BASELINE_PRECISION = "Float32"
BASELINE_COMPUTE_UNIT = "cpuOnly"


def load_timing(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_mel(precision: str, compute_unit: str) -> Optional[np.ndarray]:
    path = os.path.join(MEL_DIR, f"output_mel_{precision}_{compute_unit}.npy")
    if not os.path.isfile(path):
        return None
    return np.load(path)


def mel_diff(mel: np.ndarray, baseline: np.ndarray) -> Tuple[float, float, float]:
    """Float32_cpuOnly の出力 mel に対する L1 / L2 / コサイン類似度を返す

    フレーム数が違う場合は短い方に合わせて切り詰める（量子化で長さが変わることはほぼ無いが念のため）。
    """
    n = min(mel.shape[0], baseline.shape[0])
    a = mel[:n].flatten().astype(np.float64)
    b = baseline[:n].flatten().astype(np.float64)
    l1 = float(np.mean(np.abs(a - b)))
    l2 = float(np.sqrt(np.mean((a - b) ** 2)))
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
    cos = float(np.dot(a, b) / denom)
    return l1, l2, cos


def main() -> int:
    timing_files = sorted(glob.glob(os.path.join(TIMING_DIR, "timing_*.json")))
    if not timing_files:
        print(f"timing JSON が見つかりません: {TIMING_DIR}", file=sys.stderr)
        print("先に scripts/extract_ui_test_results.sh を実行してください", file=sys.stderr)
        return 1

    baseline_mel = load_mel(BASELINE_PRECISION, BASELINE_COMPUTE_UNIT)
    if baseline_mel is None:
        print(
            f"基準 mel ({BASELINE_PRECISION}/{BASELINE_COMPUTE_UNIT}) が見つかりません: "
            f"output_mel_{BASELINE_PRECISION}_{BASELINE_COMPUTE_UNIT}.npy",
            file=sys.stderr,
        )
        return 1

    rows = []
    for path in timing_files:
        t = load_timing(path)
        precision = t["precision"]
        compute_unit = t["computeUnit"]
        mel = load_mel(precision, compute_unit)
        if mel is None:
            l1, l2, cos = float("nan"), float("nan"), float("nan")
        else:
            l1, l2, cos = mel_diff(mel, baseline_mel)

        # 旧フォーマット (modelSizeBytes 等が無い JSON) でも落ちないように get で読む
        rows.append(
            {
                "precision": precision,
                "computeUnit": compute_unit,
                "totalMs": t.get("totalPredictMs", float("nan")),
                "rtf": t.get("realTimeFactor", float("nan")),
                "decAvgMs": t.get("decoderAvgPerStepMs", float("nan")),
                "modelMB": t.get("modelSizeBytes", 0) / (1024 * 1024) if t.get("modelSizeBytes") else float("nan"),
                "melL1": l1,
                "melL2": l2,
                "melCos": cos,
            }
        )

    # Precision (Float32, Float16, Int8) → ComputeUnit (cpuOnly, cpuAndGPU, cpuAndNE, all) の順で並べる
    precision_order = {"Float32": 0, "Float16": 1, "Int8": 2}
    compute_unit_order = {"cpuOnly": 0, "cpuAndGPU": 1, "cpuAndNE": 2, "all": 3}
    rows.sort(
        key=lambda r: (
            precision_order.get(r["precision"], 99),
            compute_unit_order.get(r["computeUnit"], 99),
        )
    )

    # 表示
    header = f"基準: {BASELINE_PRECISION}/{BASELINE_COMPUTE_UNIT} (mel L1/L2/Cos はこの mel に対する差)"
    print(header)
    print("=" * len(header))
    print(
        f"{'Precision':<8} {'Device':<10} "
        f"{'totalMs':>9} {'RTF':>6} {'dec/step':>9} {'modelMB':>8} "
        f"{'melL1':>7} {'melL2':>7} {'melCos':>7}"
    )
    print("-" * 80)
    for r in rows:
        print(
            f"{r['precision']:<8} {r['computeUnit']:<10} "
            f"{r['totalMs']:>9.0f} {r['rtf']:>6.2f} {r['decAvgMs']:>9.2f} "
            f"{r['modelMB']:>8.1f} "
            f"{r['melL1']:>7.3f} {r['melL2']:>7.3f} {r['melCos']:>7.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
