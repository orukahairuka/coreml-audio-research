#!/usr/bin/env python3
"""可変長（range1）× 精度 × Compute Units を実機で 1 構成ずつ確かめる（VariableLengthMatrixTests を回す）

1 構成ごとに別の xcodebuild test-without-building として実行するので、GPU の E5RT エラーや ANE の読み込みで
プロセスごと落ちても、他の構成は続けられる。最後に端末の結果を取り出し、PyTorch FP32（正式評価と同じ前処理）と比べる。

使い方
  PronounSE/venv/bin/python scripts/run_variable_length_matrix.py \
      --udid <xcodebuild 用の UDID> --device <devicectl 用の Identifier> \
      --conditions Float32:cpuAndGPU,Float32:cpuOnly --inputs s0p5 \
      --reference metrics/preprocess_fullinput_20261122/synth_mac_range1_f32_cpu --out metrics/variable_length_matrix_<日付>
  （--skip-build で前回のビルドを使う）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT = REPO_ROOT / "ios/CoreMLAudioApp/CoreMLAudioApp.xcodeproj"
BUNDLE_ID = "erika.com.CoreMLAudioApp"
TEST = "CoreMLAudioAppTests/VariableLengthMatrixTests"


def xcodebuild(action: str, udid: str, derived: Path, env: dict | None = None, log: Path | None = None,
               timeout: int = 1800) -> tuple[int | str, float]:
    cmd = ["xcodebuild", action, "-project", str(PROJECT), "-scheme", "CoreMLAudioApp",
           "-destination", f"id={udid}", "-derivedDataPath", str(derived), f"-only-testing:{TEST}"]
    t0 = time.time()
    try:
        with open(log or os.devnull, "w") as f:
            p = subprocess.run(cmd, env={**os.environ, **(env or {})}, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout, stdin=subprocess.DEVNULL)
        return p.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - t0


def diff(a: np.ndarray, b: np.ndarray) -> dict | None:
    if a.shape != b.shape:
        return None
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    snr = 10 * np.log10(np.sum(a64 ** 2) / max(np.sum((a64 - b64) ** 2), 1e-300))
    den = np.linalg.norm(a64) * np.linalg.norm(b64)
    return {"max_abs": float(np.abs(a64 - b64).max()), "snr_db": float(snr), "cosine": float(a64 @ b64 / den) if den else None}


def log_signature(log: Path) -> str:
    """xcodebuild のログから、失敗の手がかりになる行を拾う"""
    if not log.exists():
        return ""
    keys = ("E5RT", "Submit Async failed", "crash", "Crash", "unexpected exit", "Restarting after", "signal",
            "Issue recorded", "recorded an issue", "Caught error", "TEST FAILED", "TEST SUCCEEDED")
    lines = [ln.strip() for ln in log.read_text(errors="replace").splitlines() if any(k in ln for k in keys)]
    return " | ".join(dict.fromkeys(lines))[:1500]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--udid", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--conditions", required=True, help="Precision:CU をカンマ区切りで（実行する順）")
    ap.add_argument("--inputs", default="s0p5")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--reference-suffix", default="_22k")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    derived = args.out.parent / "_derived_data"

    if not args.skip_build:
        code, sec = xcodebuild("build-for-testing", args.udid, derived, log=args.out / "build.log")
        print(f"build-for-testing: {code}（{sec:.0f} 秒）", flush=True)
        if code != 0:
            return 1

    runs = []
    for cond in args.conditions.split(","):
        precision, cu = cond.split(":")
        label = f"{precision}_{cu}"
        env = {"TEST_RUNNER_CMLA_VLM_PRECISION": precision, "TEST_RUNNER_CMLA_VLM_CU": cu,
               "TEST_RUNNER_CMLA_VLM_INPUTS": args.inputs}
        log = args.out / f"{label}.log"
        code, sec = xcodebuild("test-without-building", args.udid, derived, env, log, args.timeout)
        text = log.read_text(errors="replace") if log.exists() else ""
        runs.append({"label": label, "exit": code, "seconds": sec, "log": log_signature(log),
                     "e5rt_lines": sum("E5RT" in ln for ln in text.splitlines()),
                     "e5rt_reshape_lines": text.count("reshaping a BNNS Op"),
                     "e5rt_no_memory_lines": text.count("No memory object bound to port")})
        print(f"{label}: exit={code}（{sec:.0f} 秒） {runs[-1]['log'][:300]}", flush=True)

    # 端末の結果を取り出す
    dev_dir = args.out / "device"
    subprocess.run(["xcrun", "devicectl", "device", "copy", "from", "--device", args.device,
                    "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE_ID,
                    "--source", "Documents/Result/variable_length_matrix", "--destination", str(dev_dir)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    table = []
    for run in runs:
        st_path = dev_dir / run["label"] / "status.json"
        st = json.loads(st_path.read_text()) if st_path.exists() else {}
        base = {"condition": run["label"], "exit": run["exit"], "wall_sec": round(run["seconds"], 1),
                "phase": st.get("phase", "no status"), "error": st.get("error", ""),
                "load_ms": st.get("load_ms"), "compute_plan": st.get("compute_plan"),
                "memory_after_load_mb": st.get("memory_after_load_mb"), "log": run["log"],
                "e5rt_lines": run["e5rt_lines"], "e5rt_reshape_lines": run["e5rt_reshape_lines"],
                "e5rt_no_memory_lines": run["e5rt_no_memory_lines"]}
        rows = st.get("rows") or [{}]
        for r in rows:
            entry = dict(base)
            entry.update({k: r.get(k) for k in ["name", "status", "T", "samples", "expected_samples", "postnet_nonfinite",
                                                 "wave_nonfinite", "raw_peak", "wave_peak", "wave_clipped_samples",
                                                 "preprocess_ms", "encoder_predict_ms", "decoder_predict_ms",
                                                 "hifigan_predict_ms", "postprocess_ms", "inference_wall_ms",
                                                 "memory_after_infer_mb", "error"] if k in r})
            n = r.get("name")
            if n and r.get("status") == "ok":
                ref = args.reference
                rn = n + args.reference_suffix
                pt_post = np.load(ref / f"{rn}_pytorch_postnet.npy")
                pt_wave = np.load(ref / f"{rn}_pytorch_wave.npy").ravel()
                dpost = diff(pt_post, np.load(dev_dir / run["label"] / f"{n}_postnet.npy"))
                dwave = diff(pt_wave, np.load(dev_dir / run["label"] / f"{n}_wave.npy").ravel())
                entry["pytorch_peak"] = float(np.abs(pt_wave).max())
                entry["pytorch_clipped_samples"] = int((np.abs(pt_wave) >= 1).sum())
                entry["postnet_snr_db"] = dpost and dpost["snr_db"]
                entry["wave_snr_db"] = dwave and dwave["snr_db"]
                entry["wave_cosine"] = dwave and dwave["cosine"]
            table.append(entry)

    (args.out / "matrix.json").write_text(json.dumps(table, ensure_ascii=False, indent=2))
    cols = ["condition", "name", "exit", "phase", "status", "T", "samples", "expected_samples", "postnet_nonfinite",
            "wave_nonfinite", "wave_peak", "pytorch_peak", "wave_clipped_samples", "pytorch_clipped_samples",
            "postnet_snr_db", "wave_snr_db", "wave_cosine", "preprocess_ms", "encoder_predict_ms", "decoder_predict_ms",
            "hifigan_predict_ms", "inference_wall_ms", "memory_after_load_mb", "memory_after_infer_mb",
            "e5rt_lines", "e5rt_reshape_lines", "e5rt_no_memory_lines", "error"]
    with open(args.out / "matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    for e in table:
        print(f"{e['condition']:<18} {str(e.get('name')):<6} exit={e['exit']} phase={e['phase']:<18} "
              f"T={e.get('T')} samples={e.get('samples')}/{e.get('expected_samples')} "
              f"NaN={e.get('postnet_nonfinite')}/{e.get('wave_nonfinite')} peak={e.get('wave_peak')} (PT {e.get('pytorch_peak')}) "
              f"clip={e.get('wave_clipped_samples')}/{e.get('pytorch_clipped_samples')} "
              f"postSNR={e.get('postnet_snr_db')} waveSNR={e.get('wave_snr_db')} "
              f"dec={e.get('decoder_predict_ms')} hifi={e.get('hifigan_predict_ms')} load={e.get('load_ms')} "
              f"E5RT={e.get('e5rt_lines')} plan={e.get('compute_plan')} err={str(e.get('error'))[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
