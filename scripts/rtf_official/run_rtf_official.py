#!/usr/bin/env python3
"""iPhone 17 Pro での正式 RTF 測定を実行する（アプリの RTFBenchRunner を起動する）

設計
  - 入力: eval/fad/data/formal_inputs/m-00/ の 100 件（22050 Hz float32 wav、FP32/FP16/INT8 で同じ入力・同じ T）
  - 構成: A=FP32（Enc GPU / Dec GPU padT / HiFi CPU）、B=FP16（Enc GPU / Dec CPU / HiFi GPU）、C=INT8（すべて CPU）
  - 入力の順番を固定シードで並べ替え、10 件ずつ 10 ブロックに分ける。各ブロックで A・B・C を 1 起動ずつ（計 30 起動）。
    構成の順番はブロックごとにずらす（ABC → BCA → CAB → …）
  - 1 起動 = 新しいプロセス。起動直後に熱状態が nominal になるまで待ち（上限 --thermal-wait）、モデルを読み込み、
    入力ごとにウォームアップ W 回 → 本測定 R 回を続けて行う
  - 起動の前に毎回、固定の休止（--rest）を入れる（最初の起動を除く）
  - 再開: 取り出した記録を見て、R 回の本測定がそろっていない入力だけを渡して起動し直す（既存の記録は消さない）

使い方
  PronounSE/venv/bin/python scripts/rtf_official/run_rtf_official.py --device <devicectl の Identifier> \
      --study rtf_official_20261123 --push
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "metrics/rtf_official"  # 記録の置き場所（metrics/ は git の対象外）
BUNDLE_ID = "erika.com.CoreMLAudioApp"
SPEAKER = "m-00"

CONFIGS = {
    "A": {"precision": "Float32", "enc": "cpuAndGPU", "dec": "cpuAndGPU", "hifi": "cpuOnly", "dec_input": "padT"},
    "B": {"precision": "Float16", "enc": "cpuAndGPU", "dec": "cpuOnly", "hifi": "cpuAndGPU", "dec_input": "grow"},
    "C": {"precision": "Int8", "enc": "cpuOnly", "dec": "cpuOnly", "hifi": "cpuOnly", "dec_input": "grow"},
}
ORDERS = ["ABC", "BCA", "CAB"]


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def devicectl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["xcrun", "devicectl", *args], capture_output=True, text=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_plan(inputs: Path, seed: int, block_size: int, warmups: int, repeats: int) -> dict:
    manifest = [r for r in csv.DictReader(open(inputs / "manifest.csv")) if r["speaker"] == SPEAKER]
    names = sorted(Path(r["output"]).stem for r in manifest)
    files = sorted(p.stem for p in (inputs / SPEAKER).glob("*.wav"))
    if names != files or len(names) != 100:
        raise SystemExit(f"m-00 の入力が台帳とそろっていない（台帳 {len(names)} 件、フォルダ {len(files)} 件）")
    order = names[:]
    random.Random(seed).shuffle(order)
    blocks = [order[i:i + block_size] for i in range(0, len(order), block_size)]
    steps = []
    for b, block in enumerate(blocks):
        for slot, label in enumerate(ORDERS[b % len(ORDERS)]):
            steps.append({"step": len(steps), "block": b, "slot_in_block": slot, "label": label, "names": block})
    return {
        "created_at": now(), "speaker": SPEAKER, "seed": seed, "block_size": block_size,
        "warmups_per_input": warmups, "repeats_per_input": repeats, "config_orders": ORDERS,
        "configs": CONFIGS, "input_order": order,
        "inputs": {Path(r["output"]).stem: {"T": int(r["expected_T"]), "samples_22050": int(r["samples_22050"]),
                                             "output_sha256": r["output_sha256"]} for r in manifest},
        "steps": steps,
    }


def record_model_hashes(path: Path, app: Path | None) -> None:
    model_dir = REPO_ROOT / "ios/CoreMLAudioApp/CoreMLAudioApp/MLModels"
    out: dict = {"recorded_at": now(), "mlpackage": {}, "compiled_in_app": {}}
    for prec in ["float32", "float16", "int8"]:
        for role in ["Transformer_Encoder", "Transformer_Decoder", "HiFiGAN_Generator"]:
            pkg = model_dir / f"{role}_{prec}_range1.mlpackage"
            out["mlpackage"][pkg.name] = {str(p.relative_to(pkg)): sha256(p) for p in sorted(pkg.rglob("*")) if p.is_file()}
            if app is not None:
                mc = app / f"{role}_{prec}_range1.mlmodelc"
                out["compiled_in_app"][mc.name] = {str(p.relative_to(mc)): sha256(p) for p in sorted(mc.rglob("*")) if p.is_file()}
    if app is not None:
        out["app_binary_sha256"] = sha256(app / "CoreMLAudioApp")
        out["app_path"] = str(app)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))


def push_inputs(device: str, inputs: Path) -> None:
    files = sorted((inputs / SPEAKER).glob("*.wav"))
    for f in files:
        r = devicectl("device", "copy", "to", "--device", device, "--domain-type", "appDataContainer",
                      "--domain-identifier", BUNDLE_ID, "--source", str(f),
                      "--destination", f"Documents/RTFInputs/{SPEAKER}/{f.name}")
        if r.returncode != 0:
            raise SystemExit(f"送れなかった: {f}\n{r.stderr[-500:]}")
    print(f"{SPEAKER}: {len(files)} 件を送った", flush=True)


def pull(device: str, study: str, dest: Path) -> bool:
    """端末の記録を取り出して dest に重ねる（起動ごとのファイルなので、既存のファイルは同じ起動のものだけが伸びる）"""
    with tempfile.TemporaryDirectory() as tmp:
        r = devicectl("device", "copy", "from", "--device", device, "--domain-type", "appDataContainer",
                      "--domain-identifier", BUNDLE_ID, "--source", f"Documents/Result/rtf_official/{study}",
                      "--destination", tmp)
        if r.returncode != 0:
            print(f"取り出せなかった: {r.stderr[-300:]}", flush=True)
            return False
        dest.mkdir(parents=True, exist_ok=True)
        for p in Path(tmp).rglob("*"):
            if p.is_file():
                shutil.copy2(p, dest / p.name)
    return True


def timed_ok_by_launch(device_dir: Path) -> dict[tuple[str, str], int]:
    """(起動, 入力) ごとの本測定 ok の回数"""
    counts: dict[tuple[str, str], int] = {}
    for f in device_dir.glob("iter_*.csv"):
        for r in csv.DictReader(open(f)):
            if r["phase"] == "timed" and r["status"] == "ok":
                key = (r["launch"], r["input"])
                counts[key] = counts.get(key, 0) + 1
    return counts


def remaining_names(step: dict, device_dir: Path, repeats: int) -> list[str]:
    counts = timed_ok_by_launch(device_dir)
    complete = {name for (launch, name), n in counts.items()
                if launch.startswith(f"{step['label']}_b{step['block']:02d}_") and n >= repeats}
    return [n for n in step["names"] if n not in complete]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--inputs", type=Path, default=REPO_ROOT / "eval/fad/data/formal_inputs")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--app", type=Path, help="インストールしたアプリ（.app）。コンパイル済みモデルのハッシュを記録する")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--block-size", type=int, default=10)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--rest", type=float, default=180, help="起動の前の休止（秒）")
    ap.add_argument("--thermal-wait", type=int, default=900, help="起動直後に nominal を待つ上限（秒）")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--steps", help="実行するステップ番号（動作確認用、カンマ区切り）")
    ap.add_argument("--limit-names", type=int, help="各ステップの先頭から何件まで（動作確認用）")
    ap.add_argument("--timeout", type=float, default=3 * 3600, help="1 起動の上限（秒）")
    args = ap.parse_args()

    study_dir = OUT_ROOT / args.study
    device_dir = study_dir / "device"
    study_dir.mkdir(parents=True, exist_ok=True)
    plan_path = study_dir / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = make_plan(args.inputs, args.seed, args.block_size, args.warmups, args.repeats)
        plan["rest_s"] = args.rest
        plan["thermal_wait_limit_s"] = args.thermal_wait
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    if not (study_dir / "model_hashes.json").exists():
        record_model_hashes(study_dir / "model_hashes.json", args.app)
    sessions = study_dir / "sessions.jsonl"

    def log(event: dict) -> None:
        event = {"at": now(), **event}
        with open(sessions, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)

    if args.push:
        push_inputs(args.device, args.inputs)
        log({"event": "push", "count": 100})

    selected = None if args.steps is None else {int(s) for s in args.steps.split(",")}
    first = True
    for step in plan["steps"]:
        if selected is not None and step["step"] not in selected:
            continue
        for attempt in range(args.max_attempts):
            todo = remaining_names(step, device_dir, plan["repeats_per_input"])
            if args.limit_names:
                todo = [n for n in todo if n in step["names"][:args.limit_names]]
            if not todo:
                break
            if not first and args.rest > 0:
                log({"event": "rest", "seconds": args.rest, "before_step": step["step"]})
                time.sleep(args.rest)
            first = False
            cfg = CONFIGS[step["label"]]
            launch = f"{step['label']}_b{step['block']:02d}_{time.strftime('%Y%m%d%H%M%S')}"
            env = {
                "CMLA_RTF_BENCH": "1", "CMLA_RTF_STUDY": args.study, "CMLA_RTF_LAUNCH": launch,
                "CMLA_RTF_LABEL": step["label"], "CMLA_RTF_PRECISION": cfg["precision"],
                "CMLA_RTF_CU_ENC": cfg["enc"], "CMLA_RTF_CU_DEC": cfg["dec"], "CMLA_RTF_CU_HIFI": cfg["hifi"],
                "CMLA_RTF_DEC_INPUT": cfg["dec_input"], "CMLA_RTF_SPEAKER": SPEAKER, "CMLA_RTF_NAMES": ",".join(todo),
                "CMLA_RTF_WARMUPS": str(plan["warmups_per_input"]), "CMLA_RTF_REPEATS": str(plan["repeats_per_input"]),
                "CMLA_RTF_THERMAL_WAIT_S": str(plan["thermal_wait_limit_s"]),
            }
            log({"event": "launch", "step": step["step"], "block": step["block"], "slot": step["slot_in_block"],
                 "label": step["label"], "launch": launch, "attempt": attempt, "names": len(todo)})
            console = study_dir / "console" / f"{launch}.log"
            console.parent.mkdir(exist_ok=True)
            started = time.monotonic()
            with open(console, "w") as out:
                proc = subprocess.Popen(["xcrun", "devicectl", "device", "process", "launch", "--device", args.device,
                                         "--console", "--terminate-existing", "--environment-variables",
                                         json.dumps(env), BUNDLE_ID], stdout=out, stderr=subprocess.STDOUT)
                try:
                    proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
            text = console.read_text(errors="replace")
            done = [ln for ln in text.splitlines() if "RTF_BENCH_DONE" in ln]
            status = done[-1].strip() if done else "no RTF_BENCH_DONE (crash or launch failure)"
            pulled = pull(args.device, args.study, device_dir)
            log({"event": "finished", "launch": launch, "seconds": round(time.monotonic() - started, 1),
                 "status": status, "pulled": pulled})
        else:
            left = remaining_names(step, device_dir, plan["repeats_per_input"])
            if left:
                log({"event": "gave_up", "step": step["step"], "remaining": left})
    log({"event": "all_done"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
