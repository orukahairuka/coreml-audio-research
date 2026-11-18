#!/usr/bin/env python3
"""高速化の各段階の前後比較を集計する。

同じ入力・同じ回の A と B の差（B - A）を対にして、中央値と 95% ブートストラップ信頼区間を求める。
時間と出力の基準（最終的な採否は、これに加えて画面の応答などへの影響も考えて決める）:
  1. A と B で、入力メル・Decoder の出力・HiFi-GAN の出力・波形の SHA-256 がすべて一致する
  2. モデルの呼び出し以外の時間の差の信頼区間の上限が 0 未満である（確かに短くなった）
  3. 合成処理時間の差の中央値が 0 未満である（モデルの呼び出しの増加で短縮が打ち消されていない）
syncPrediction はモデルの呼び出しの方法の変更なので、2 と 3 の代わりに、
合成処理時間の差の信頼区間の上限が 0 未満であることを基準にする。

実行例:
  PronounSE/venv/bin/python scripts/aggregate_optimization_ladder.py metrics/optimization_20261118/iPhone_13
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HASHES = ["mel_in_sha256", "decoder_mel_sha256", "hifigan_wave_sha256", "deemph_wave_sha256"]
METRICS = {"e2e": "e2e_ms", "predict": "predict_sum_ms", "non_predict": "non_predict_ms"}


def interval(values: np.ndarray, seed: int = 20261118, count: int = 10000) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    picks = generator.integers(0, len(values), size=(count, len(values)))
    medians = np.median(values[picks], axis=1)
    return float(np.median(values)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def analyze(directory: Path) -> dict:
    path = next(directory.rglob("optab_runs.csv"))
    rows = list(csv.DictReader(open(path, newline="")))
    pairs: dict = defaultdict(dict)
    for row in rows:
        pairs[(row["precision"], row["compute_units"], row["input_name"], row["round"])][row["variant"]] = row
    complete = [p for p in pairs.values() if len(p) == 2 and all(r["status"] == "success" for r in p.values())]
    hash_ok = all(p["A"][k] == p["B"][k] for p in complete for k in HASHES)
    hash_kinds = {name: len({r["deemph_wave_sha256"] for r in rows if r["input_name"] == name and r["status"] == "success"})
                  for name in {r["input_name"] for r in rows}}
    result: dict = {
        "step": directory.name, "variant_a": rows[0]["optimizations"] if rows[0]["variant"] == "A" else rows[1]["optimizations"],
        "variant_b": rows[0]["optimizations"] if rows[0]["variant"] == "B" else rows[1]["optimizations"],
        "pairs": len(complete), "expected_pairs": len(pairs), "failed_runs": sum(r["status"] != "success" for r in rows),
        "thermal": sorted({r["thermal_after"] for r in rows}), "outputs_identical": hash_ok, "waveform_hashes_per_input": hash_kinds,
    }
    for key, column in METRICS.items():
        a = np.array([float(p["A"][column]) for p in complete])
        b = np.array([float(p["B"][column]) for p in complete])
        median, low, high = interval(b - a)
        result[key] = {"a_median_ms": float(np.median(a)), "b_median_ms": float(np.median(b)),
                       "delta_median_ms": median, "delta_ci95_ms": [low, high]}
    if directory.name.endswith("syncPrediction"):
        faster = result["e2e"]["delta_ci95_ms"][1] < 0
    else:
        faster = result["non_predict"]["delta_ci95_ms"][1] < 0 and result["e2e"]["delta_median_ms"] < 0
    result["meets_criteria"] = bool(hash_ok and result["failed_runs"] == 0 and faster)
    conditions: dict = defaultdict(lambda: {"pairs": 0, "identical": 0})
    for p in complete:
        entry = conditions[f"{p['A']['precision']}:{p['A']['compute_units']}"]
        entry["pairs"] += 1
        entry["identical"] += int(all(p["A"][k] == p["B"][k] for k in HASHES))
    result["by_condition"] = dict(conditions)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = [analyze(d) for d in sorted(p for p in args.root.iterdir() if p.is_dir() and list(p.rglob("optab_runs.csv")))]
    (args.root / "ladder_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for r in report:
        e, p, n = r["e2e"], r["predict"], r["non_predict"]
        fmt = lambda m: f"{m['delta_median_ms']:+8.2f} [{m['delta_ci95_ms'][0]:+7.2f},{m['delta_ci95_ms'][1]:+7.2f}]"
        print(f"{r['step']:24s} n={r['pairs']:2d} 出力一致={r['outputs_identical']} 熱={','.join(r['thermal'])} "
              f"E2E {e['a_median_ms']:7.1f}→{e['b_median_ms']:7.1f} 差{fmt(e)} 推論差{fmt(p)} 推論外差{fmt(n)} 基準={r['meets_criteria']}")
        if len(r["by_condition"]) > 1:
            for name, entry in sorted(r["by_condition"].items()):
                print(f"    {name:42s} 出力が一致した対 {entry['identical']}/{entry['pairs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
