#!/usr/bin/env python3
"""range1 の GPU・ANE 実行の問題を切り分ける（VariableLengthDiagnosticsTests を 1 構成ずつ実機で回す）

構成は「ラベル:変数=値;変数=値」をカンマ区切りで渡す（変数名は CMLA_VLD_ を省いたもの）。例:
  A_f32:PRECISION=Float32;CU_ENC=cpuAndGPU;CU_DEC=cpuOnly;CU_HIFI=cpuAndGPU
最後に端末の結果を取り出し、PyTorch FP32（正式評価と同じ前処理）と比べ、--baseline の構成と出力がビット一致するかも調べる
（ビット一致なら、その構成は baseline と同じ計算をした可能性が高い。実行先の推定の手がかりにする）。

使い方
  PronounSE/venv/bin/python scripts/run_variable_length_diagnostics.py --udid <UDID> --device <Identifier> \
      --conditions '<ラベル>:<変数>=<値>;...' --inputs s0p5,s3 \
      --reference metrics/preprocess_fullinput_20261122/synth_mac_range1_f32_cpu --out metrics/<フォルダ>
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import librosa
import numpy as np

from run_variable_length_matrix import BUNDLE_ID, diff, log_signature, xcodebuild

TEST = "CoreMLAudioAppTests/VariableLengthDiagnosticsTests"


def logmel(x: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=22050, n_fft=1024, hop_length=256, n_mels=80)
    return 10 * np.log10(np.maximum(m, 1e-10))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--udid", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--inputs", default="s0p5")
    ap.add_argument("--passes", default="2")
    ap.add_argument("--mem-limit-mb", default="1500")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--reference-suffix", default="_22k")
    ap.add_argument("--baseline", type=Path, help="ビット一致を調べる相手の構成のフォルダ（例: 前回の F32 cpuOnly の device/<label>）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    derived = args.out.parent / "_derived_data"

    import run_variable_length_matrix as m
    m.TEST = TEST  # xcodebuild の -only-testing をこのテストにする
    if not args.skip_build:
        code, sec = xcodebuild("build-for-testing", args.udid, derived, log=args.out / "build.log")
        print(f"build-for-testing: {code}（{sec:.0f} 秒）", flush=True)
        if code != 0:
            return 1

    runs = []
    for spec in args.conditions.split(","):
        label, _, rest = spec.partition(":")
        env = {"TEST_RUNNER_CMLA_VLD_LABEL": label, "TEST_RUNNER_CMLA_VLD_INPUTS": args.inputs,
               "TEST_RUNNER_CMLA_VLD_PASSES": args.passes, "TEST_RUNNER_CMLA_VLD_MEM_LIMIT_MB": args.mem_limit_mb}
        for kv in filter(None, rest.split(";")):
            k, _, v = kv.partition("=")
            env[f"TEST_RUNNER_CMLA_VLD_{k}"] = v
        log = args.out / f"{label}.log"
        code, sec = xcodebuild("test-without-building", args.udid, derived, env, log, args.timeout)
        text = log.read_text(errors="replace") if log.exists() else ""
        runs.append({"label": label, "exit": code, "seconds": sec, "log": log_signature(log),
                     "crashed": "Restarting after unexpected exit" in text,
                     "e5rt_lines": sum("E5RT" in ln for ln in text.splitlines())})
        print(f"{label}: exit={code}（{sec:.0f} 秒） crashed={runs[-1]['crashed']}", flush=True)

    dev = args.out / "device"
    subprocess.run(["xcrun", "devicectl", "device", "copy", "from", "--device", args.device,
                    "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE_ID,
                    "--source", "Documents/Result/variable_length_diag", "--destination", str(dev)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    table = []
    for run in runs:
        st_path = dev / run["label"] / "status.json"
        st = json.loads(st_path.read_text()) if st_path.exists() else {}
        for r in st.get("rows") or [{}]:
            e = {"label": run["label"], "exit": run["exit"], "crashed": run["crashed"], "e5rt_lines": run["e5rt_lines"],
                 "phase": st.get("phase", "no status"), "units": st.get("compute_units"), "plan": st.get("compute_plan"),
                 "load_ms": st.get("load_ms"), "decoder_input": st.get("decoder_input"),
                 "reshape": st.get("reshape_frequency"), "hifigan_model": (st.get("models") or {}).get("hifigan"),
                 "memory_after_release_mb": st.get("memory_after_release_mb"),
                 "reloaded": {k: (st.get("reloaded_row") or {}).get(k)
                              for k in ["status", "reload_ms", "decoder_ms", "memory_after_mb"]},
                 "load_error": st.get("error", "")}
            e.update({k: r.get(k) for k in ["name", "pass", "status", "T", "samples", "expected_samples",
                                             "postnet_nonfinite", "wave_nonfinite", "wave_peak", "wave_clipped_samples",
                                             "encoder_ms", "decoder_ms", "hifigan_ms", "memory_before_mb",
                                             "memory_after_decoder_mb", "memory_after_mb", "error"]})
            tr = r.get("decoder_memory_trace") or []
            if tr:
                e["decoder_mem_first_last"] = [tr[0][1], tr[-1][1]]
                e["decoder_mem_growth_mb_per_100_steps"] = (tr[-1][1] - tr[0][1]) / max(tr[-1][0] - tr[0][0], 1) * 100
            n = r.get("name")
            if n and r.get("status") == "ok" and r.get("pass") == 1:
                rn = n + args.reference_suffix
                pt_post = np.load(args.reference / f"{rn}_pytorch_postnet.npy")
                pt_wave = np.load(args.reference / f"{rn}_pytorch_wave.npy").ravel()
                post = np.load(dev / run["label"] / f"{n}_postnet.npy")
                wave = np.load(dev / run["label"] / f"{n}_wave.npy").ravel()
                d = diff(pt_post, post)
                e["postnet_snr_db"] = d and d["snr_db"]
                e["logmel_mae_db"] = float(np.abs(logmel(wave) - logmel(pt_wave)).mean()) if wave.size == pt_wave.size else None
                e["pytorch_peak"] = float(np.abs(pt_wave).max())
                if args.baseline is not None and (args.baseline / f"{n}_wave.npy").exists():
                    e["bit_identical_to_baseline"] = bool(np.array_equal(np.load(args.baseline / f"{n}_wave.npy").ravel(), wave))
            table.append(e)

    (args.out / "diagnostics.json").write_text(json.dumps(table, ensure_ascii=False, indent=2))
    cols = ["label", "name", "pass", "status", "exit", "crashed", "T", "samples", "expected_samples", "postnet_nonfinite",
            "wave_nonfinite", "wave_peak", "pytorch_peak", "wave_clipped_samples", "postnet_snr_db", "logmel_mae_db",
            "bit_identical_to_baseline", "encoder_ms", "decoder_ms", "hifigan_ms", "memory_before_mb",
            "memory_after_decoder_mb", "memory_after_mb", "decoder_mem_growth_mb_per_100_steps", "e5rt_lines", "error",
            "load_error"]
    with open(args.out / "diagnostics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    for e in table:
        rnd = lambda v: round(v, 1) if isinstance(v, float) else v  # noqa: E731
        print(f"{e['label']:<16} {str(e.get('name')):<5} p{e.get('pass')} {str(e.get('status')):<8} crash={e['crashed']} "
              f"T={e.get('T')} len={e.get('samples')}/{e.get('expected_samples')} NaN={e.get('postnet_nonfinite')}/{e.get('wave_nonfinite')} "
              f"peak={rnd(e.get('wave_peak'))}/PT{rnd(e.get('pytorch_peak'))} postSNR={rnd(e.get('postnet_snr_db'))} "
              f"logmel={rnd(e.get('logmel_mae_db'))} bit={e.get('bit_identical_to_baseline')} "
              f"enc={rnd(e.get('encoder_ms'))} dec={rnd(e.get('decoder_ms'))} hifi={rnd(e.get('hifigan_ms'))} "
              f"mem={rnd(e.get('memory_before_mb'))}->{rnd(e.get('memory_after_decoder_mb'))}->{rnd(e.get('memory_after_mb'))} "
              f"grow/100={rnd(e.get('decoder_mem_growth_mb_per_100_steps'))} E5RT={e['e5rt_lines']} "
              f"err={str(e.get('error') or e.get('load_error'))[:160]}")
    for run in runs:
        st_path = dev / run["label"] / "status.json"
        if st_path.exists():
            st = json.loads(st_path.read_text())
            print(f"  {run['label']}: plan={st.get('compute_plan')} load={st.get('load_ms')} "
                  f"release={st.get('memory_after_release_mb')} reloaded={st.get('reloaded_row', {}).get('status')} "
                  f"reload_ms={st.get('reloaded_row', {}).get('reload_ms')} reload_dec={st.get('reloaded_row', {}).get('decoder_ms')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
