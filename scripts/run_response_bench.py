#!/usr/bin/env python3
"""通常操作の応答時間（合成の開始から再生開始の API 呼び出しまで）を実機で測る。

アプリを CMLA_RESPONSE_BENCH=1 で起動すると、通常の画面（ContentView）のまま
SynthesisViewModel.runSynthesis() を入力ごとに「ウォームアップ → 本計測」の順で繰り返し、
1 回ごとに Documents/Result/response/<session>/response_runs.csv に 1 行を書いて終了する。
合成・保存・再生の処理経路は画面の「合成」ボタンと同じ（合成音は端末のスピーカーから再生される）。

計測を始める前に、卒論第4章の計測と同じ方法（CMLA_THESIS_BENCH=thermal で起動して読む）で
thermalState が --thermal-ok-states になるまで待つ。

出力（--out-dir、既定 metrics/response_<端末>_<日時>/raw/）:
  host_runs.csv       起動ごとの結果
  thermal_checks.csv  thermal の確認ごとの結果
  logs/<launch>.log   起動ごとのコンソール出力
  device/             アプリの Documents/Result/response/<session>/ を吸い出したもの

実行例:
  # iPhone 17 Pro（計測していない状態でも fair から下がらないので fair まで許す）
  scripts/run_response_bench.py --device "櫻井絵理香のiPhone" --install path/to/CoreMLAudioApp.app \\
      --thermal-ok-states nominal,fair
  # iPhone 13
  scripts/run_response_bench.py --device "iPhone (3)" --install path/to/CoreMLAudioApp.app
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_thesis_bench import REPO_ROOT, Target, append_csv, now_iso  # noqa: E402

BENCH_VERSION = "resp-v1"  # アプリ側の ResponseBenchConfig.benchVersion と同じ

HOST_HEADER = [
    "session_id", "launch_id", "mode", "precision", "compute_setting", "warmup", "steady", "inputs",
    "gap_ms", "thermal_wait_sec", "thermal_first_check", "thermal_at_start", "started_at", "ended_at",
    "elapsed_sec", "status", "exit_code", "log_path", "bench_version",
]
THERMAL_HEADER = ["session_id", "checked_at", "before_launch_id", "state", "status"]


class ResponseBench:
    def __init__(self, args: argparse.Namespace, target: Target):
        self.args = args
        self.target = target
        self.out_dir: Path = args.out_dir
        self.logs = self.out_dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def next_launch_id(self, mode: str) -> str:
        self.counter += 1
        return f"L{self.counter:04d}_{mode}"

    def launch(self, env: dict[str, str], launch_id: str, timeout: float,
               done_marker: str, start_marker: str) -> tuple[str, int | None, str, float]:
        import os
        import subprocess
        cmd, extra_env = self.target.launch_cmd(env)
        log_path = self.logs / f"{launch_id}.log"
        started = time.monotonic()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **extra_env})
            try:
                exit_code = proc.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                exit_code, timed_out = None, True
        elapsed = time.monotonic() - started
        text = log_path.read_text(errors="replace")
        if timed_out:
            self.target.terminate()
            status = "timeout"
        elif f"{done_marker} status=ok" in text:
            status = "ok"
        elif f"{done_marker} status=" in text:
            status = "failed"
        elif start_marker not in text:
            status = "launch_failed"
        else:
            status = "crashed"
        return status, exit_code, text, elapsed

    def check_thermal(self, before_launch_id: str) -> str:
        launch_id = self.next_launch_id("thermal")
        env = {"CMLA_THESIS_BENCH": "thermal", "CMLA_THESIS_SESSION": self.args.session,
               "CMLA_THESIS_LAUNCH": launch_id}
        status, _, text, _ = self.launch(env, launch_id, 120, "THESIS_BENCH_DONE", "THESIS_BENCH_START")
        m = re.search(r"THESIS_THERMAL=(\w+)", text)
        state = m.group(1) if m else "unknown"
        append_csv(self.out_dir / "thermal_checks.csv", THERMAL_HEADER, {
            "session_id": self.args.session, "checked_at": now_iso(),
            "before_launch_id": before_launch_id, "state": state, "status": status,
        })
        return state

    def wait_thermal(self, before_launch_id: str) -> tuple[float, str, str]:
        started = time.monotonic()
        first = None
        unknown = 0
        while True:
            state = self.check_thermal(before_launch_id)
            unknown = unknown + 1 if state == "unknown" else 0
            if unknown >= 3:
                raise SystemExit("thermal を 3 回続けて読めなかった。logs/ を確認してください")
            if first is None:
                first = state
                if state in self.args.thermal_ok_states:
                    return 0.0, first, state
            waited = time.monotonic() - started
            if state in self.args.thermal_ok_states:
                return waited, first, state
            if waited > self.args.thermal_max_wait:
                print(f"  thermal が {self.args.thermal_max_wait}s 待っても {state} のまま。そのまま進める")
                return waited, first, state
            print(f"  thermal={state} → {self.args.thermal_poll}s 待つ")
            time.sleep(self.args.thermal_poll)

    def run(self) -> int:
        if self.args.install:
            if not self.target.install(Path(self.args.install)):
                raise SystemExit("インストールに失敗しました")
        launch_id = self.next_launch_id("response")
        wait, first, state = (0.0, "", "")
        if not self.args.no_thermal_wait:
            wait, first, state = self.wait_thermal(launch_id)
        env = {
            "CMLA_RESPONSE_BENCH": "1", "CMLA_RESPONSE_SESSION": self.args.session,
            "CMLA_RESPONSE_LAUNCH": launch_id, "CMLA_RESPONSE_PRECISION": self.args.precision,
            "CMLA_RESPONSE_SETTING": self.args.setting,
            "CMLA_RESPONSE_WARMUP": str(self.args.warmup), "CMLA_RESPONSE_STEADY": str(self.args.steady),
            "CMLA_RESPONSE_GAP_MS": str(self.args.gap_ms),
        }
        if self.args.inputs:
            env["CMLA_RESPONSE_INPUTS"] = self.args.inputs
        print(f"[{now_iso()}] {launch_id} {self.args.precision}:{self.args.setting} (thermal wait {wait:.0f}s, {state})")
        started_at = now_iso()
        status, exit_code, _, elapsed = self.launch(env, launch_id, self.args.timeout,
                                                     "RESPONSE_BENCH_DONE", "RESPONSE_BENCH_START")
        print(f"  → {status} ({elapsed:.1f}s)")
        append_csv(self.out_dir / "host_runs.csv", HOST_HEADER, {
            "session_id": self.args.session, "launch_id": launch_id, "mode": "response",
            "precision": self.args.precision, "compute_setting": self.args.setting,
            "warmup": self.args.warmup, "steady": self.args.steady, "inputs": self.args.inputs or "",
            "gap_ms": self.args.gap_ms, "thermal_wait_sec": f"{wait:.1f}", "thermal_first_check": first,
            "thermal_at_start": state, "started_at": started_at, "ended_at": now_iso(),
            "elapsed_sec": f"{elapsed:.1f}", "status": status,
            "exit_code": "" if exit_code is None else exit_code, "log_path": f"logs/{launch_id}.log",
            "bench_version": BENCH_VERSION,
        })
        print("===== 結果を吸い出す =====")
        ok = self.target.copy_from_documents(f"Result/response/{self.args.session}", self.out_dir / "device")
        print("  → " + ("OK " + str(self.out_dir / "device") if ok else "失敗（手で copy from してください）"))
        return 0 if status == "ok" and ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--device", help="devicectl の名前")
    target.add_argument("--simulator", help="シミュレータの UDID（スクリプトの動作確認用。計測結果には使わない）")
    p.add_argument("--install", help="インストールする .app（Release ビルド）")
    p.add_argument("--precision", default="Float16")
    p.add_argument("--setting", default="cpuAndNE+hifi-cpuOnly", help="画面の Compute Units の rawValue（既定は条件 C）")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steady", type=int, default=3)
    p.add_argument("--inputs", help="入力の名前をカンマ区切りで（既定は 1,6,11,2 ＝入力1〜4）")
    p.add_argument("--gap-ms", type=int, default=1000, help="再生終了から次の合成までの待ち")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--thermal-ok-states", default="nominal",
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    p.add_argument("--thermal-poll", type=float, default=60)
    p.add_argument("--thermal-max-wait", type=float, default=1800)
    p.add_argument("--no-thermal-wait", action="store_true")
    p.add_argument("--session")
    p.add_argument("--out-dir", type=Path)
    args = p.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = re.sub(r"[^\w]+", "_", args.device or f"sim_{args.simulator}").strip("_")
    args.session = args.session or f"response_{stamp}"
    args.out_dir = args.out_dir or REPO_ROOT / "metrics" / f"response_{label}_{stamp}" / "raw"
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"{args.out_dir} は空ではありません（既存の記録を上書きしないため止めます）")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return ResponseBench(args, Target(args.device, args.simulator)).run()


if __name__ == "__main__":
    sys.exit(main())
