#!/usr/bin/env python3
"""通常操作の応答時間の計測（scripts/run_response_bench.py）の結果を集計する。

入力ごとに本計測（phase=steady）の中央値・最小・最大を出す。ウォームアップは集計に入れない。
--pipeline に卒論第4章の計測用パイプラインの infer_runs.csv（最終の実装・区間記録なし）を渡すと、
同じ入力の End-to-End 時間の中央値と、出力波形（デエンファシス後）の SHA-256 の一致を並べて出す。

実行例:
  scripts/aggregate_response_bench.py metrics/response_iPhone_17_Pro_<日時> \\
      --pipeline metrics/thesis_ch4v4_decinput_iPhone_17_Pro_20261118_001315/bulk_noprofile/raw/device/infer_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

# 論文での呼び方（入力1〜4）と同梱ファイルの名前
INPUT_LABEL = {"1": "入力1", "6": "入力2", "11": "入力3", "2": "入力4"}
METRICS = ["A_to_B_ms", "B_to_C_ms", "C_to_D_ms", "A_to_D_ms", "D_to_E_ms", "E_to_F_ms", "F_to_G_ms",
           "A_to_F_ms", "A_to_G_ms", "synth_ms"]


def summarize(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{st.median(values):.1f} [{min(values):.1f}–{max(values):.1f}]"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir", type=Path, help="metrics/response_<端末>_<日時>")
    p.add_argument("--pipeline", type=Path, help="比較する計測用パイプラインの infer_runs.csv")
    args = p.parse_args()

    csv_path = next(args.session_dir.rglob("response_runs.csv"), None)
    if csv_path is None:
        print("response_runs.csv が見つかりません", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(open(csv_path)))
    r0 = rows[0]
    print(f"# {args.session_dir}")
    print(f"device={r0['device_model']} os={r0['os_version']} build={r0['build_configuration']} "
          f"precision={r0['precision']} setting={r0['compute_setting']} bench={r0['bench_version']}")
    print(f"rows={len(rows)} success={sum(r['status'] == 'success' for r in rows)} "
          f"thermal_before={sorted({r['thermal_before'] for r in rows})} "
          f"thermal_after={sorted({r['thermal_after'] for r in rows})}")

    pipe: dict[str, list[dict]] = {}
    if args.pipeline:
        for r in csv.DictReader(open(args.pipeline)):
            if r["phase"] == "steady":
                pipe.setdefault(r["input_name"], []).append(r)

    order = sorted({(int(r["input_order"]), r["input_name"]) for r in rows})
    print("\n## 本計測の中央値 [最小–最大] (ms)")
    print("| 入力 | n | 合成音の長さ [s] | " + " | ".join(METRICS) + (" | パイプラインのE2E | 波形の一致 |" if pipe else " |"))
    for _, name in order:
        steady = [r for r in rows if r["input_name"] == name and r["phase"] == "steady" and r["status"] == "success"]
        cells = [summarize([float(r[m]) for r in steady if r[m]]) for m in METRICS]
        audio = steady[0]["audio_sec"] if steady else "-"
        line = f"| {INPUT_LABEL.get(name, name)} | {len(steady)} | {float(audio):.2f} | " + " | ".join(cells)
        if pipe:
            pr = pipe.get(name, [])
            e2e = st.median(float(r["e2e_ms"]) for r in pr) if pr else float("nan")
            hashes = {r["output_sha256"] for r in rows if r["input_name"] == name}
            pipe_hashes = {r["deemph_wave_sha256"] for r in pr}
            same = "一致" if hashes and hashes == pipe_hashes else f"不一致 {len(hashes)}/{len(pipe_hashes)}"
            line += f" | {e2e:.1f} | {same} |"
        else:
            line += " |"
        print(line)

    print("\n## 1 回ごと")
    for r in rows:
        print(f"{INPUT_LABEL.get(r['input_name'], r['input_name'])} #{r['run_index']} {r['phase']:<6} {r['status']} "
              f"{r['thermal_before']}→{r['thermal_after']} A→B={r['A_to_B_ms']} A→F={r['A_to_F_ms']} "
              f"synth={r['synth_ms']} sha={r['output_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
