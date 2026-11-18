#!/usr/bin/env python3
"""合成処理時間（End-to-End）の処理別の内訳を集計する。

run_thesis_bench.py --profile の infer_stages.csv と infer_runs.csv の定常時（phase=steady）を使う。
区間は隙間なく並ぶので、1 回の合成の全区間の合計は e2e_ms と一致する。

実行例:
  PronounSE/venv/bin/python scripts/aggregate_e2e_stages.py metrics/e2e_analysis_20261118/iPhone_13
  # T' = 262 の 2 入力だけを全体の表に使う
  PronounSE/venv/bin/python scripts/aggregate_e2e_stages.py metrics/e2e_analysis_20261118/iPhone_13_reversed --inputs 1,6
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

INPUTS = {"1": "入力1", "6": "入力2", "11": "入力3", "2": "入力4"}

# 表の行: (表示名, 区間, モデルの呼び出しか)
GROUPS = [
    ("前処理: 読み込み・リサンプリング", ["pre_audio_load"], False),
    ("前処理: メルスペクトログラム", ["pre_mel_extract"], False),
    ("Encoder: 入力準備", ["enc_input_fill", "enc_input_validate", "enc_pos_fill", "enc_feature_provider"], False),
    ("Encoder: 推論", ["enc_predict"], True),
    ("Encoder: 出力の取り出し", ["enc_output_extract"], False),
    ("Decoder: 入力配列の確保と0埋め", ["dec_input_alloc"], False),
    ("Decoder: 入力配列への書き込み", ["dec_input_fill"], False),
    ("Decoder: 入力の検証", ["dec_input_validate"], False),
    ("Decoder: posの作成", ["dec_pos_fill"], False),
    ("Decoder: 入力辞書の作成", ["dec_feature_provider"], False),
    ("Decoder: 繰り返し推論", ["dec_predict"], True),
    ("Decoder: 出力の取り出し", ["dec_output_extract"], False),
    ("Decoder: 次の入力の読み取り", ["dec_autoregress_read"], False),
    ("Decoder: 進捗通知", ["dec_step_stats", "dec_progress_yield"], False),
    ("HiFi-GAN: 入力準備", ["voc_input_transpose", "voc_input_validate", "voc_feature_provider"], False),
    ("HiFi-GAN: 推論", ["voc_predict"], True),
    ("HiFi-GAN: 出力の取り出し", ["voc_output_read"], False),
    ("デエンファシス", ["post_deemph"], False),
]


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row["phase"] == "steady"]


def summarize(rows: list[dict]) -> dict:
    e2e = [float(r["e2e_ms"]) for r in rows]
    total = sum(e2e)
    table = []
    for label, stages, is_model in GROUPS:
        values = [sum(float(r[f"{s}_ms"]) for s in stages) for r in rows]
        table.append({"stage": label, "model_call": is_model, "median_ms": st.median(values),
                      "share_pct": sum(values) / total * 100})
    predict = [float(r["predict_sum_ms"]) for r in rows]
    other = [float(r["non_predict_ms"]) for r in rows]
    return {"n": len(rows), "e2e_median_ms": st.median(e2e), "predict_median_ms": st.median(predict),
            "non_predict_median_ms": st.median(other), "predict_share_pct": sum(predict) / total * 100,
            "non_predict_share_pct": sum(other) / total * 100, "stages": table}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--inputs", help="全体の表に使う入力の名前（カンマ区切り。既定は全入力）")
    args = parser.parse_args()
    selected_inputs = set(args.inputs.split(",")) if args.inputs else set(INPUTS)
    device = args.root / "profile/raw/device"
    stages = read(device / "infer_stages.csv")
    runs = read(device / "infer_runs.csv")
    plain = read(args.root / "noprofile/raw/device/infer_runs.csv")

    report: dict = {"checks": {
        "steady_runs": len(stages), "failed": sum(r["status"] != "success" for r in runs + plain),
        "thermal": sorted({r["thermal_after"] for r in runs + plain}),
        "max_abs_profile_residual_ms": max(abs(float(r["profile_residual_ms"])) for r in stages),
        "bench_version": sorted({r["bench_version"] for r in runs + plain}),
    }}
    report["selected_inputs"] = sorted(selected_inputs)
    report["all_inputs"] = summarize([r for r in stages if r["input_name"] in selected_inputs])
    report["by_input"] = {}
    for name, label in INPUTS.items():
        selected = [r for r in stages if r["input_name"] == name]
        run_rows = [r for r in runs if r["input_name"] == name]
        plain_rows = [r for r in plain if r["input_name"] == name]
        entry = summarize(selected)
        steps = int(selected[0]["decoder_steps"])
        step_ms = [np.load(device / f"{r['file_prefix']}_decoder_step_ms.npy").ravel() for r in run_rows]
        joined = np.concatenate(step_ms)
        entry.update({
            "T": int(run_rows[0]["T"]), "T_prime": int(run_rows[0]["T_prime"]), "decoder_steps": steps,
            "audio_sec": float(run_rows[0]["audio_sec"]),
            "e2e_noprofile_median_ms": st.median(float(r["e2e_ms"]) for r in plain_rows),
            "synth_noprofile_median_ms": st.median(float(r["synth_ms"]) for r in plain_rows),
            "decoder_predict_step_ms": {
                "median": float(np.median(joined)), "p95": float(np.percentile(joined, 95)),
                "max": float(joined.max()), "first_step_median": float(np.median([s[0] for s in step_ms])),
            },
            "output_hash_distinct": len({r["deemph_wave_sha256"] for r in run_rows + plain_rows}),
            "mel_hash_distinct": len({r["mel_in_sha256"] for r in run_rows + plain_rows}),
        })
        report["by_input"][label] = entry
    report["all_inputs"]["e2e_noprofile_median_ms"] = st.median(
        float(r["e2e_ms"]) for r in plain if r["input_name"] in selected_inputs
    )
    suffix = "" if not args.inputs else "_in" + "_".join(sorted(selected_inputs))
    (args.root / f"e2e_report{suffix}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    a = report["all_inputs"]
    print(json.dumps(report["checks"], ensure_ascii=False))
    print(f"E2E 中央値 {a['e2e_median_ms']:.1f} ms（記録なし {a['e2e_noprofile_median_ms']:.1f} ms） "
          f"モデル呼び出し {a['predict_median_ms']:.1f} ms（{a['predict_share_pct']:.1f}%） "
          f"それ以外 {a['non_predict_median_ms']:.1f} ms（{a['non_predict_share_pct']:.1f}%）")
    for s in sorted(a["stages"], key=lambda x: -x["share_pct"]):
        print(f"  {s['stage']:<24} {s['median_ms']:9.2f} ms {s['share_pct']:6.2f}%{'  [推論]' if s['model_call'] else ''}")
    for label, e in report["by_input"].items():
        d = e["decoder_predict_step_ms"]
        print(f"{label}: T={e['T']} T'={e['T_prime']} 音声{e['audio_sec']:.2f}s E2E={e['e2e_median_ms']:.0f}/記録なし{e['e2e_noprofile_median_ms']:.0f} ms "
              f"推論外={e['non_predict_median_ms']:.1f} ms Decoder1回 中央値{d['median']:.2f} p95 {d['p95']:.2f} 最大{d['max']:.2f} 初回{d['first_step_median']:.2f} ms "
              f"波形hash種類={e['output_hash_distinct']} メルhash種類={e['mel_hash_distinct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
