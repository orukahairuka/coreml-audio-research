#!/usr/bin/env python3
"""正式評価（FAD・RTF）用の一括合成を実機で行う（アプリの FormalEvalRunner を起動する）

手順
  1. --push: 入力（scripts/prepare_formal_eval_inputs.py が作った 22050 Hz の float32 の wav）を
     端末の Documents/FormalEvalInputs/<話者>/ に送る
  2. CMLA_FORMAL_EVAL=1 でアプリを起動し、終わるまで待つ（devicectl --console のログに FORMAL_EVAL_DONE が出る）
  3. 端末の Documents/Result/formal_eval/<session>/ を取り出す
  4. ホストで確かめる: 出力の長さが T × 256、wav（PCM16）と npy（float32）が対応している、
     出力の名前から参照音（Trimmed_Explosion/<参照>.wav）がたどれる、results.csv の行と出力がそろっている

アプリは事前にインストールしておく（Xcode か xcodebuild でこのブランチのものを入れる）。
同じ --session で起動し直すと、results.csv で ok の入力は飛ばして続きから合成する（既存の出力は上書きしない）。

使い方（動作確認の例）
  PronounSE/venv/bin/python scripts/run_formal_eval.py --device <devicectl の Identifier> \
      --session smoke_20261122 --precision Float32 --cu cpuOnly --speakers m-01 --limit 2 --push \
      --reference-dir ~/Desktop/Trimmed_Explosion --out metrics/formal_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ID = "erika.com.CoreMLAudioApp"


def devicectl(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["xcrun", "devicectl", *args], capture_output=capture, text=True)


def push_inputs(device: str, inputs: Path, speakers: list[str], names: set[str] | None, limit: int | None) -> None:
    """端末に送る。アプリと同じく、話者ごとにファイル名の順で --names と --limit を当てたものだけを送る"""
    for spk in speakers:
        files = sorted((inputs / spk).glob("*.wav"))
        if names is not None:
            files = [f for f in files if f.stem in names]
        if limit is not None:
            files = files[:limit]
        for f in files:
            r = devicectl("device", "copy", "to", "--device", device, "--domain-type", "appDataContainer",
                          "--domain-identifier", BUNDLE_ID, "--source", str(f),
                          "--destination", f"Documents/FormalEvalInputs/{spk}/{f.name}")
            if r.returncode != 0:
                raise SystemExit(f"送れなかった: {f}\n{r.stderr[-500:]}")
        print(f"  {spk}: {len(files)} 件を送った", flush=True)


def record_run_config(out_dir: Path, args: argparse.Namespace, env: dict, speakers: list[str],
                      names: set[str] | None) -> None:
    """起動ごとに、条件・入力の一覧・モデルのファイルのハッシュ・Git のコミットを run_config.json に追記する"""
    import datetime as dt
    import hashlib

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    model_dir = REPO_ROOT / "ios/CoreMLAudioApp/CoreMLAudioApp/MLModels"
    models = {}
    for role in ["Transformer_Encoder", "Transformer_Decoder", "HiFiGAN_Generator"]:
        pkg = model_dir / f"{role}_{args.precision.lower()}_range1.mlpackage"
        models[pkg.name] = {str(p.relative_to(pkg)): sha256(p) for p in sorted(pkg.rglob("*")) if p.is_file()}
    manifest = {r["output"]: r for r in csv.DictReader(open(args.inputs / "manifest.csv"))}
    inputs = []
    for spk in speakers:
        files = sorted((args.inputs / spk).glob("*.wav"))
        if names is not None:
            files = [f for f in files if f.stem in names]
        if args.limit:
            files = files[:args.limit]
        for f in files:
            m = manifest.get(f"{spk}/{f.name}", {})
            inputs.append({"input": f"{spk}/{f.name}", "source_sha256": m.get("source_sha256"),
                           "output_sha256": m.get("output_sha256"), "expected_T": m.get("expected_T")})
    git = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    record = {"launched_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "env": env,
              "git_commit": git, "git_dirty": bool(dirty.strip()), "models": models,
              "inputs_count": len(inputs), "inputs": inputs}
    path = out_dir / "run_config.json"
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(record)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2))


def verify(session_dir: Path, inputs: Path, reference_dir: Path | None) -> dict:
    """端末から取り出した結果を確かめる"""
    rows = list(csv.DictReader(open(session_dir / "results.csv"))) if (session_dir / "results.csv").exists() else []
    manifest = {r["output"]: r for r in csv.DictReader(open(inputs / "manifest.csv"))}
    problems = []
    ok = [r for r in rows if r["status"] == "ok"]
    for r in ok:
        wav, npy = session_dir / r["output_wav"], session_dir / r["output_f32"]
        if not wav.exists() or not npy.exists():
            problems.append(f"{r['output_wav']}: 出力のファイルが無い")
            continue
        m = manifest.get(r["input_file"])
        if m is None:
            problems.append(f"{r['input_file']}: 入力の台帳に無い")
        elif int(m["expected_T"]) != int(r["T"]) or int(m["samples_22050"]) != int(r["input_samples"]):
            problems.append(f"{r['input_file']}: T か入力のサンプル数が台帳と違う")
        pcm, sr = sf.read(wav, dtype="int16")
        f32 = np.load(npy).ravel()
        info = sf.info(wav)
        if sr != 22050 or info.subtype != "PCM_16" or info.channels != 1:
            problems.append(f"{r['output_wav']}: 形式が 22050 Hz・モノラル・PCM_16 でない")
        if len(pcm) != int(r["T"]) * 256 or f32.size != len(pcm):
            problems.append(f"{r['output_wav']}: 長さが T × 256 と違う")
        else:
            # PCM16 は float32 を ±1 で切って量子化したもの（丸めの違いは 1 LSB まで）
            expect = np.clip(f32, -1, 1) * 32768
            if np.abs(pcm.astype(np.float64) - expect).max() > 1.5:
                problems.append(f"{r['output_wav']}: wav と npy が対応しない")
        expected_name = f"{r['reference']}_{r['speaker']}_pronounse_ios.wav"
        if Path(r["output_wav"]).name != expected_name:
            problems.append(f"{r['output_wav']}: 名前が {expected_name} でない")
        if reference_dir is not None and not (reference_dir / f"{r['reference']}.wav").exists():
            problems.append(f"{r['output_wav']}: 参照音 {r['reference']}.wav が無い")
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {"rows": len(rows), "status_counts": statuses, "verified_ok": len(ok), "problems": problems}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="devicectl の Identifier か名前")
    ap.add_argument("--session", required=True)
    ap.add_argument("--precision", required=True, choices=["Float32", "Float16", "Int8"])
    ap.add_argument("--cu", required=True, choices=["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"])
    ap.add_argument("--speakers", default="f-01,m-01,m-02,m-03,m-04,m-05")
    ap.add_argument("--names", help="<参照>_<話者> をカンマ区切りで")
    ap.add_argument("--limit", type=int, help="話者ごとに先頭から何件まで（動作確認用）")
    ap.add_argument("--inputs", type=Path, default=REPO_ROOT / "eval/fad/data/formal_inputs")
    ap.add_argument("--push", action="store_true", help="入力を端末に送る")
    ap.add_argument("--reference-dir", type=Path, help="参照音のフォルダ（出力の名前から参照音をたどれるか確かめる）")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "metrics/formal_eval")
    ap.add_argument("--timeout", type=float, default=6 * 3600)
    exp = ap.add_argument_group("実験用（GPU・ANE の切り分け。指定しなければ従来と同じ）")
    exp.add_argument("--cu-enc", choices=["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"])
    exp.add_argument("--cu-dec", choices=["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"])
    exp.add_argument("--cu-hifi", choices=["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"])
    exp.add_argument("--dec-input", choices=["grow", "padT"], help="padT: Decoder 入力を毎回 T フレームにする")
    exp.add_argument("--reload-mb", type=float, help="メモリがこれを超えたらモデルを読み込み直す")
    args = ap.parse_args()
    speakers = args.speakers.split(",")
    names = set(args.names.split(",")) if args.names else None
    out_dir = args.out / args.session
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.push:
        print("入力を送る", flush=True)
        push_inputs(args.device, args.inputs, speakers, names, args.limit)

    env = {"CMLA_FORMAL_EVAL": "1", "CMLA_FORMAL_SESSION": args.session, "CMLA_FORMAL_PRECISION": args.precision,
           "CMLA_FORMAL_CU": args.cu, "CMLA_FORMAL_SPEAKERS": ",".join(speakers)}
    if names:
        env["CMLA_FORMAL_NAMES"] = ",".join(sorted(names))
    if args.limit:
        env["CMLA_FORMAL_LIMIT"] = str(args.limit)
    for key, value in [("CMLA_FORMAL_CU_ENC", args.cu_enc), ("CMLA_FORMAL_CU_DEC", args.cu_dec),
                       ("CMLA_FORMAL_CU_HIFI", args.cu_hifi), ("CMLA_FORMAL_DEC_INPUT", args.dec_input),
                       ("CMLA_FORMAL_RELOAD_MB", None if args.reload_mb is None else str(args.reload_mb))]:
        if value:
            env[key] = value
    record_run_config(out_dir, args, env, speakers, names)
    log_path = out_dir / f"launch_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"起動 {env}", flush=True)
    started = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(["xcrun", "devicectl", "device", "process", "launch", "--device", args.device,
                                 "--console", "--terminate-existing", "--environment-variables", json.dumps(env),
                                 BUNDLE_ID], stdout=log, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
    text = log_path.read_text(errors="replace")
    done = [ln for ln in text.splitlines() if "FORMAL_EVAL_DONE" in ln]
    status = done[-1] if done else ("FORMAL_EVAL_START はあるが終わりが無い（落ちた可能性）"
                                    if "FORMAL_EVAL_START" in text else "起動できなかった")
    print(f"終了（{time.monotonic() - started:.0f} 秒）: {status}", flush=True)

    device_dir = out_dir / "device"
    r = devicectl("device", "copy", "from", "--device", args.device, "--domain-type", "appDataContainer",
                  "--domain-identifier", BUNDLE_ID, "--source", f"Documents/Result/formal_eval/{args.session}",
                  "--destination", str(device_dir))
    if r.returncode != 0:
        print(f"取り出せなかった: {r.stderr[-300:]}")
        return 1
    report = verify(device_dir, args.inputs, args.reference_dir)
    report["launch_status"] = status
    (out_dir / "verify.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["problems"] and done and "status=ok" in done[-1] else 2


if __name__ == "__main__":
    sys.exit(main())
