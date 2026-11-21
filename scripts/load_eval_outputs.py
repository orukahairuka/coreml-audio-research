#!/usr/bin/env python3
"""計測モード（infer / vocoder）が実行ごとに保存した配列を、meta.json を手がかりに読み込む。

アプリは実行ごとに `runs/<launch>/<prefix>_meta.json` を書き、同じ prefix の npy と wav の
形状・データ型・サンプリング周波数・実験条件をそこに記録する（ThesisRunMetadata.swift）。
このスクリプトは、その meta.json を探して一覧にし、配列を読み込むだけである。比較の計算はしない。

使い方:
  # 一覧（入力・条件・回・T′・各配列の形状）
  scripts/load_eval_outputs.py metrics/<dir>/raw/device
  # 絞り込み
  scripts/load_eval_outputs.py metrics/<dir>/raw/device --input 1 --precision Float16 --phase steady

Python から:
  from load_eval_outputs import find_runs, load_arrays
  for run in find_runs(Path("metrics/<dir>/raw/device"), input_name="1"):
      arrays = load_arrays(run)          # {"decoder_mel": (T', 256), "hifigan": (256T',), "deemph": (256T',), ...}
      sr = run.meta["audio"]["sample_rate"]

波形（hifigan, deemph）は npy では (1, N) で保存されている。load_arrays は既定で (N,) にして返す。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalRun:
    meta_path: Path
    meta: dict

    @property
    def prefix(self) -> str:
        return self.meta_path.name[: -len("_meta.json")]

    @property
    def input_name(self) -> str:
        return self.meta["input"]["name"]

    @property
    def condition(self) -> str:
        return self.meta["condition_label"]

    @property
    def device(self) -> str:
        return self.meta["device"]["model"]

    def path(self, key: str) -> Path:
        return self.meta_path.parent / self.meta["arrays"][key]["file"]


def find_runs(root: Path, *, input_name: str | None = None, precision: str | None = None,
              condition: str | None = None, phase: str | None = None, mode: str | None = None) -> list[EvalRun]:
    """root の下の *_meta.json を集める（並びは条件・入力・回の順）"""
    runs = []
    for meta_path in sorted(root.rglob("*_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "arrays" not in meta or "format_version" not in meta:
            continue
        run = EvalRun(meta_path, meta)
        if input_name is not None and run.input_name != input_name:
            continue
        if precision is not None and meta.get("precision") != precision:
            continue
        if condition is not None and run.condition != condition:
            continue
        if phase is not None and meta["run"].get("phase") != phase:
            continue
        if mode is not None and meta.get("mode") != mode:
            continue
        runs.append(run)
    runs.sort(key=lambda r: (r.device, r.meta.get("mode", ""), r.condition, r.input_name, r.meta["run"]["run_index"]))
    return runs


def _read_wav_int16(path: Path) -> tuple[np.ndarray, int]:
    import wave
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: 16 ビット・モノラルではない")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return data, w.getframerate()


def load_arrays(run: EvalRun, keys: list[str] | None = None, *, squeeze: bool = True,
                verify: bool = True) -> dict[str, np.ndarray]:
    """meta.json に書かれた配列を読み込む。

    verify=True のときは、形状とデータ型が meta.json と一致するか（wav はサンプリング周波数も）を確かめる。
    squeeze=True のときは、波形の (1, N) を (N,) にする。wav は int16 のまま返す（/32768 はしない）。
    """
    out = {}
    for key, spec in run.meta["arrays"].items():
        if keys is not None and key not in keys:
            continue
        path = run.path(key)
        if spec["format"] == "wav":
            arr, sr = _read_wav_int16(path)
            if verify and sr != run.meta["audio"]["sample_rate"]:
                raise ValueError(f"{path}: サンプリング周波数 {sr} が meta.json と違う")
        else:
            arr = np.load(path)
        if verify:
            if list(arr.shape) != list(spec["shape"]):
                raise ValueError(f"{path}: 形状 {arr.shape} が meta.json の {spec['shape']} と違う")
            if arr.dtype != np.dtype(spec["dtype"]):
                raise ValueError(f"{path}: データ型 {arr.dtype} が meta.json の {spec['dtype']} と違う")
        if squeeze and arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        out[key] = arr
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="device/ のディレクトリ（またはその上）")
    parser.add_argument("--input")
    parser.add_argument("--precision")
    parser.add_argument("--condition")
    parser.add_argument("--phase", choices=["initial", "warmup", "steady"])
    parser.add_argument("--mode", choices=["infer", "vocoder"])
    parser.add_argument("--verify", action="store_true", help="配列を読み込み、形状とデータ型を meta.json と照合する")
    args = parser.parse_args()

    runs = find_runs(args.root, input_name=args.input, precision=args.precision,
                     condition=args.condition, phase=args.phase, mode=args.mode)
    if not runs:
        print("meta.json が見つからない", file=sys.stderr)
        return 1
    print("device\tos\tmode\tcondition\tinput\trun\tphase\tT\tT'\tsr\tarrays")
    for run in runs:
        m = run.meta
        shapes = " ".join(f"{k}{tuple(v['shape'])}:{v['dtype']}" for k, v in sorted(m["arrays"].items()))
        print("\t".join(str(x) for x in [
            run.device, m["device"]["os_version"], m["mode"], run.condition, run.input_name,
            m["run"]["run_index"], m["run"]["phase"], m["input"]["T"], m["input"]["T_prime"],
            m["audio"]["sample_rate"], shapes,
        ]))
        if args.verify:
            load_arrays(run)
    if args.verify:
        print(f"# {len(runs)} 回分の配列の形状とデータ型が meta.json と一致した")
    return 0


if __name__ == "__main__":
    sys.exit(main())
