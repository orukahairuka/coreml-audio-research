#!/usr/bin/env python3
"""卒論第4章の計測を実機（またはシミュレータ）で回す。

アプリを計測モード（環境変数 CMLA_THESIS_BENCH）で起動し直しながら、
load → plan → infer → vocoder の順に 12 条件を測る。
load は条件ごとに「インストールし直してから起動」を 3 回（インストール直後の初回の読み込み）と
「起動し直すだけ」を 2 回（Core ML がディスクに残した準備結果を再利用する初回の読み込み）測る。1 回の起動ごとに host_runs.csv に 1 行を書き、
アプリが THESIS_BENCH_DONE を出さずに終わった場合は crashed、時間内に終わらなければ timeout とする。
条件を変える前に thermalState が nominal に戻るまで待ち、待った秒数も記録する。
異常終了・時間切れがあっても、条件を外したりやり直したりはしない（そのまま次へ進む）。

出力（--out-dir、既定 metrics/thesis_ch4_<端末>_<日時>/raw/）:
  host_runs.csv        起動ごとの結果（status, 経過秒, thermal の待ち時間, crumb）
  thermal_checks.csv   thermal の確認ごとの結果
  logs/<launch>.log    起動ごとのコンソール出力
  device/              アプリの Documents/Result/thesis/<session>/ を吸い出したもの

実行例:
  # リハーサル: 1 条件 × 1 入力
  scripts/run_thesis_bench.py --device "iPhone (3)" --phases infer \\
      --conditions Float32:cpuOnly --inputs 1
  # 本番（iPhone 13）
  scripts/run_thesis_bench.py --device "iPhone (3)" --install path/to/CoreMLAudioApp.app
  # シミュレータでスクリプトの動作確認
  scripts/run_thesis_bench.py --simulator <UDID> --phases infer --conditions Float32:cpuOnly --inputs 1
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ID = "erika.com.CoreMLAudioApp"

PRECISIONS = ["Float32", "Float16", "Int8"]
# ANE を使う条件（cpuAndNE, all）を後に回す
COMPUTE_UNITS = ["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"]
ALL_CONDITIONS = [(p, cu) for cu in COMPUTE_UNITS for p in PRECISIONS]

DEFAULT_TIMEOUT = {"load": 600, "plan": 600, "infer": 1800, "vocoder": 900, "thermal": 120}

HOST_HEADER = [
    "session_id", "launch_id", "mode", "precision", "compute_units", "launch_index",
    "load_kind", "after_install", "thermal_wait_sec", "thermal_at_start", "started_at", "ended_at",
    "elapsed_sec", "status", "exit_code", "crumb", "log_path",
]
THERMAL_HEADER = ["session_id", "checked_at", "before_launch_id", "state", "status"]


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def append_csv(path: Path, header: list[str], row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


class Target:
    """実機（devicectl）かシミュレータ（simctl）か"""

    def __init__(self, device: str | None, simulator: str | None):
        self.device = device
        self.simulator = simulator

    @property
    def label(self) -> str:
        return self.device or f"sim_{self.simulator}"

    def launch_cmd(self, env: dict[str, str]) -> tuple[list[str], dict]:
        if self.device:
            return ([
                "xcrun", "devicectl", "device", "process", "launch",
                "--device", self.device, "--console", "--terminate-existing",
                "--environment-variables", json.dumps(env), BUNDLE_ID,
            ], {})
        # simctl は SIMCTL_CHILD_ を付けた環境変数をアプリに渡す
        child_env = {f"SIMCTL_CHILD_{k}": v for k, v in env.items()}
        return ([
            "xcrun", "simctl", "launch", "--console-pty", "--terminate-running-process",
            self.simulator, BUNDLE_ID,
        ], child_env)

    def terminate(self) -> None:
        """時間切れのアプリを止める（thermal モードで起動し直すと --terminate-existing で前のプロセスが落ちる）"""
        if self.simulator:
            subprocess.run(["xcrun", "simctl", "terminate", self.simulator, BUNDLE_ID],
                           capture_output=True)

    def install(self, app_path: Path, attempts: int = 3) -> bool:
        """インストールする。devicectl はまれに即座に失敗するので数回やり直す。成功したら True"""
        if self.device:
            cmd = ["xcrun", "devicectl", "device", "install", "app", "--device", self.device, str(app_path)]
        else:
            cmd = ["xcrun", "simctl", "install", self.simulator, str(app_path)]
        for i in range(attempts):
            r = subprocess.run(cmd, capture_output=True, text=True)
            ok = r.returncode == 0 and ("installationURL" in r.stdout or self.simulator)
            if ok:
                return True
            print(f"  インストール失敗（{i + 1}/{attempts}）: {(r.stderr or r.stdout).strip()[-200:]}")
            time.sleep(5)
        return False

    def copy_from_documents(self, rel_path: str, dest: Path) -> bool:
        """Documents/<rel_path>（ディレクトリかファイル）を dest にコピーする"""
        dest.mkdir(parents=True, exist_ok=True)
        if self.device:
            r = subprocess.run([
                "xcrun", "devicectl", "device", "copy", "from", "--device", self.device,
                "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE_ID,
                "--source", f"/Documents/{rel_path}", "--destination", str(dest),
            ], capture_output=True, text=True)
            return r.returncode == 0
        r = subprocess.run(["xcrun", "simctl", "get_app_container", self.simulator, BUNDLE_ID, "data"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False
        src = Path(r.stdout.strip()) / "Documents" / rel_path
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dest / src.name)
        else:
            return False
        return True


class Bench:
    def __init__(self, args: argparse.Namespace, target: Target):
        self.args = args
        self.target = target
        self.session = args.session
        self.out_dir: Path = args.out_dir
        self.logs = self.out_dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.host_csv = self.out_dir / "host_runs.csv"
        self.thermal_csv = self.out_dir / "thermal_checks.csv"
        # 同じセッションを続きから回すとき、起動 ID（＝端末側のファイルの置き場所）が重ならないようにずらす
        self.launch_counter = args.launch_start - 1
        self.after_install = bool(args.install)

    def next_launch_id(self, mode: str) -> str:
        self.launch_counter += 1
        return f"L{self.launch_counter:04d}_{mode}"

    def launch(self, env: dict[str, str], launch_id: str, timeout: float) -> tuple[str, int | None, str, float]:
        """1 回起動して終わるのを待つ。(status, exit_code, ログ, 経過秒) を返す"""
        cmd, extra_env = self.target.launch_cmd(env)
        log_path = self.logs / f"{launch_id}.log"
        started = time.monotonic()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, **extra_env})
            try:
                exit_code = proc.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                exit_code = None
                timed_out = True
        elapsed = time.monotonic() - started
        text = log_path.read_text(errors="replace")
        if timed_out:
            self.target.terminate()
            status = "timeout"
        elif "THESIS_BENCH_DONE status=ok" in text:
            status = "ok"
        elif "THESIS_BENCH_DONE status=" in text:
            status = "failed"
        else:
            status = "crashed"
        return status, exit_code, text, elapsed

    def check_thermal(self, before_launch_id: str) -> str:
        launch_id = self.next_launch_id("thermal")
        env = {"CMLA_THESIS_BENCH": "thermal", "CMLA_THESIS_SESSION": self.session,
               "CMLA_THESIS_LAUNCH": launch_id}
        status, _, text, _ = self.launch(env, launch_id, DEFAULT_TIMEOUT["thermal"])
        m = re.search(r"THESIS_THERMAL=(\w+)", text)
        state = m.group(1) if m else "unknown"
        append_csv(self.thermal_csv, THERMAL_HEADER, {
            "session_id": self.session, "checked_at": now_iso(),
            "before_launch_id": before_launch_id, "state": state, "status": status,
        })
        return state

    def wait_nominal(self, before_launch_id: str) -> tuple[float, str]:
        """nominal に戻るまで待つ。(待った秒数, 最初に確認した状態) を返す"""
        started = time.monotonic()
        first = None
        unknown_streak = 0
        while True:
            state = self.check_thermal(before_launch_id)
            # thermal を読めないのは起動自体の失敗（署名・ロック・切断など）なので、待ち続けずに止める
            unknown_streak = unknown_streak + 1 if state == "unknown" else 0
            if unknown_streak >= 3:
                raise SystemExit(f"thermal を 3 回続けて読めなかった（アプリを起動できていない）。"
                                 f"logs/ の thermal のログを確認してください（{before_launch_id} の前）")
            if first is None:
                first = state
                if state == "nominal":
                    # 最初の確認で nominal なら待っていない（確認の起動にかかった時間は含めない）
                    return 0.0, first
            waited = time.monotonic() - started
            if state == "nominal":
                return waited, first
            if waited > self.args.thermal_max_wait:
                print(f"  thermal が {self.args.thermal_max_wait}s 待っても {state} のまま。そのまま進める")
                return waited, first
            print(f"  thermal={state} → {self.args.thermal_poll}s 待つ")
            time.sleep(self.args.thermal_poll)

    def fetch_crumb(self, launch_id: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            rel = f"Result/thesis/{self.session}/crumb_{launch_id}.txt"
            if not self.target.copy_from_documents(rel, Path(tmp)):
                return ""
            files = list(Path(tmp).rglob("crumb_*.txt"))
            return files[0].read_text().strip() if files else ""

    def run_one(self, mode: str, precision: str, cu: str, launch_index: int, wait_thermal: bool,
                load_kind: str = "") -> None:
        launch_id = self.next_launch_id(mode)
        thermal_wait, thermal_state = (0.0, "")
        if wait_thermal:
            thermal_wait, thermal_state = self.wait_nominal(launch_id)
        if load_kind == "after_install":
            # thermal の確認でアプリを起動するので、インストールはその後（起動の直前）に行う
            self.after_install = self.target.install(Path(self.args.install))
            if not self.after_install:
                load_kind = "install_failed"
        env = {
            "CMLA_THESIS_BENCH": mode, "CMLA_THESIS_SESSION": self.session,
            "CMLA_THESIS_LAUNCH": launch_id, "CMLA_THESIS_PRECISION": precision, "CMLA_THESIS_CU": cu,
            "CMLA_THESIS_RELOADS": str(self.args.reloads),
            "CMLA_THESIS_WARMUP": str(self.args.warmup), "CMLA_THESIS_STEADY": str(self.args.steady),
        }
        if self.args.inputs:
            env["CMLA_THESIS_INPUTS"] = self.args.inputs
        print(f"[{now_iso()}] {launch_id} {mode} {precision}:{cu} #{launch_index} (thermal wait {thermal_wait:.0f}s)")
        started_at = now_iso()
        status, exit_code, _, elapsed = self.launch(env, launch_id, self.args.timeout.get(mode, 1800))
        crumb = "" if status == "ok" else self.fetch_crumb(launch_id)
        print(f"  → {status} ({elapsed:.1f}s){'  crumb: ' + crumb if crumb else ''}")
        append_csv(self.host_csv, HOST_HEADER, {
            "session_id": self.session, "launch_id": launch_id, "mode": mode,
            "precision": precision, "compute_units": cu, "launch_index": launch_index,
            "load_kind": load_kind, "after_install": int(self.after_install), "thermal_wait_sec": f"{thermal_wait:.1f}",
            "thermal_at_start": thermal_state, "started_at": started_at, "ended_at": now_iso(),
            "elapsed_sec": f"{elapsed:.1f}", "status": status,
            "exit_code": "" if exit_code is None else exit_code, "crumb": crumb,
            "log_path": f"logs/{launch_id}.log",
        })
        self.after_install = False

    def run(self) -> None:
        if "load" in self.args.phases and self.args.load_install_launches > 0 and not self.args.install:
            raise SystemExit("load でインストール直後を測るには --install が必要です")
        if self.args.install and "load" not in self.args.phases:
            if not self.target.install(Path(self.args.install)):
                raise SystemExit("インストールに失敗しました")
            self.after_install = True
        for mode in self.args.phases:
            print(f"===== {mode} =====")
            for precision, cu in self.args.conditions:
                if mode == "load":
                    # インストール直後の初回 × N（毎回入れ直す）→ 起動し直すだけの初回 × M
                    kinds = (["after_install"] * self.args.load_install_launches
                             + ["restart"] * self.args.load_restart_launches)
                else:
                    kinds = [""]
                for i, kind in enumerate(kinds, start=1):
                    # 条件が変わるときだけ thermal を待つ（同じ条件の load の起動し直しの間は待たない）
                    self.run_one(mode, precision, cu, i, wait_thermal=(i == 1 and not self.args.no_thermal_wait),
                                 load_kind=kind)
        print("===== 結果を吸い出す =====")
        ok = self.target.copy_from_documents(f"Result/thesis/{self.session}", self.out_dir / "device")
        print("  → " + ("OK " + str(self.out_dir / "device") if ok else "失敗（手で copy from してください）"))


def parse_conditions(text: str | None) -> list[tuple[str, str]]:
    if not text:
        return ALL_CONDITIONS
    out = []
    for item in text.split(","):
        p, cu = item.split(":")
        if p not in PRECISIONS or cu not in COMPUTE_UNITS:
            raise SystemExit(f"不明な条件: {item}")
        out.append((p, cu))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--device", help="devicectl の名前（例: 'iPhone (3)'）")
    target.add_argument("--simulator", help="シミュレータの UDID（スクリプトの動作確認用）")
    parser.add_argument("--phases", default="load,plan,infer,vocoder")
    parser.add_argument("--conditions", help="Float32:cpuOnly,... 既定は 12 条件（ANE を使う条件が後）")
    parser.add_argument("--inputs", help="入力の名前をカンマ区切りで（既定はアプリの thesis_inputs.json の順）")
    parser.add_argument("--load-install-launches", type=int, default=3,
                        help="load で条件ごとに「インストールし直してから起動」する回数")
    parser.add_argument("--load-restart-launches", type=int, default=2,
                        help="load で条件ごとに、その後「起動し直すだけ」の回数")
    parser.add_argument("--reloads", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steady", type=int, default=5)
    parser.add_argument("--thermal-poll", type=float, default=30)
    parser.add_argument("--thermal-max-wait", type=float, default=1800)
    parser.add_argument("--no-thermal-wait", action="store_true")
    parser.add_argument("--install", help="計測の前にインストールする .app のパス")
    parser.add_argument("--session", help="セッション ID（既定は日時）。続きから回すときは前回と同じ ID を渡す")
    parser.add_argument("--launch-start", type=int, default=1,
                        help="起動 ID の開始番号。同じセッションを続きから回すときは前回より大きい番号にする")
    parser.add_argument("--out-dir", type=Path)
    for mode, sec in DEFAULT_TIMEOUT.items():
        parser.add_argument(f"--timeout-{mode}", type=float, default=sec)
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    t = Target(args.device, args.simulator)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", t.label).strip("_")
    args.session = args.session or f"{slug}_{stamp}"
    args.out_dir = args.out_dir or REPO_ROOT / "metrics" / f"thesis_ch4_{slug}_{stamp}" / "raw"
    args.phases = [p for p in args.phases.split(",") if p]
    args.conditions = parse_conditions(args.conditions)
    args.timeout = {m: getattr(args, f"timeout_{m}") for m in DEFAULT_TIMEOUT}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"bench_args_{stamp}.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2, ensure_ascii=False))
    print(f"session={args.session}  out={args.out_dir}")
    Bench(args, t).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
