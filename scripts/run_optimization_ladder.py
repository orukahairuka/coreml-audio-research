#!/usr/bin/env python3
"""高速化を 1 つずつ足しながら、変更の前後を実機で比べる。

各段階で「それまでに足した高速化（A）」と「さらに 1 つ足した高速化（B）」を、同じ起動の中で交互に測る
（アプリの optab モード）。結果は <out>/<番号>_<名前>/raw/ に保存する。

実行例:
  python3 scripts/run_optimization_ladder.py --device "iPhone (3)" --install <app> --out metrics/optimization_20261118/iPhone_13
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS = ["vocoderTranspose", "decoderFrameRead", "sparseMelFilterbank", "vocoderOutputRead",
         "encoderInputCopy", "reuseDecoderPos", "reuseDecoderInput", "syncPrediction"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", required=True)
    parser.add_argument("--install", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--steps", default=",".join(STEPS), help="試す高速化（カンマ区切り）")
    parser.add_argument("--base", default="", help="最初から有効にする高速化（カンマ区切り）")
    parser.add_argument("--thermal-ok-states", default="nominal,fair")
    args = parser.parse_args()

    applied = [x for x in args.base.split(",") if x]
    installed = False
    for index, step in enumerate(x for x in args.steps.split(",") if x):
        variant_a = ",".join(applied) or "none"
        variant_b = ",".join(applied + [step])
        slug = f"{index + 1:02d}_{step}"
        session = f"optab_{args.out.name}_{slug}"
        command = [
            sys.executable, "-u", str(REPO_ROOT / "scripts/run_thesis_bench.py"), "--device", args.device,
            "--phases", "optab", "--conditions", "Float16:cpuAndNE:cpuOnly", "--steady", str(args.rounds),
            "--opt-a", variant_a, "--opt-b", variant_b, "--session", session,
            "--thermal-ok-states", args.thermal_ok_states, "--out-dir", str(args.out / slug / "raw"),
        ]
        if not installed:
            command += ["--install", args.install]
            installed = True
        print(f"=== {slug}: A={variant_a} / B={variant_b}", flush=True)
        result = subprocess.run(command, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            return result.returncode
        applied.append(step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
