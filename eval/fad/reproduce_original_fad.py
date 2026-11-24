#!/usr/bin/env python3
"""開発者が報告したオリジナル PronounSE の PANNs-FAD（話者別 6 条件 + Whole）を、どの計算条件で再現できるかを調べる

提供されたオリジナルの合成音（PronounSE/<話者>/*.wav、600 件）と参照音（Trimmed_Explosion、100 件）から、
次の組み合わせで FAD を計算し、開発者の値と並べる。値を合わせるための補正はしない。
  - PANNs のモデル: wavegram-logmel（本実装の既定）/ cnn14-32k / cnn14-16k（kadtk の PANNsModel）
  - 標本化周波数の変換: via22050（本実装。22050 Hz を経由してモデルの周波数へ）/
                         direct（kadtk の標準の流れに近い。モデルの周波数へ直接変換し、16 ビットに量子化する）
  - 長さ: 切らない（参照音と合成音は時間で対応しないため）
  - FAD: 本実装（float64、不偏共分散、特異値による平方根の項）と、kadtk の既定（float32）
FAD の計算は compute_fad.py の frechet_distance と kadtk_frechet_distance をそのまま使う（compute_fad.py は変えない）。

出力: --out に summary.json / summary.csv、埋め込みの npz

使い方
  .venv/bin/python reproduce_original_fad.py --dataset ~/Desktop --out results/original_fad_repro
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import compute_fad as cf

SPEAKERS = ["m-01", "m-02", "m-03", "m-04", "m-05", "f-01"]
DEVELOPER = {"m-01": 17.85, "m-02": 17.05, "m-03": 17.28, "m-04": 16.65, "m-05": 19.29, "f-01": 19.28, "Whole": 14.09}


def load_model(variant: str):
    from kadtk.model_loader import PANNsModel
    ml = PANNsModel(variant)
    ml.device = torch.device("cpu")
    original_load = torch.load
    torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": "cpu"})
    try:
        ml.load_model()
    finally:
        torch.load = original_load
    return ml


def prepare(path: Path, model_sr: int, route: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    x = audio.mean(axis=1)
    if route == "via22050":
        if sr != 22050:
            x = librosa.resample(x, orig_sr=sr, target_sr=22050, res_type="soxr_hq")
        x = librosa.resample(x, orig_sr=22050, target_sr=model_sr, res_type="soxr_hq")
    else:  # direct: モデルの周波数へ直接変換し、kadtk と同じく 16 ビットに量子化する
        if sr != model_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=model_sr, res_type="soxr_hq")
        x = np.clip(np.round(x * 32768.0), -32768, 32767) / 32768.0
    return x.astype(np.float32)


def embed(ml, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        e = ml._get_embedding(x)
    return e.detach().cpu().numpy().astype(np.float32)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", default="wavegram-logmel,cnn14-32k,cnn14-16k")
    ap.add_argument("--routes", default="via22050,direct")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    ref_files = sorted((args.dataset / "Trimmed_Explosion").glob("*.wav"))
    syn_files = {s: sorted((args.dataset / "PronounSE" / s).glob("*.wav")) for s in SPEAKERS}
    if len(ref_files) != 100 or any(len(v) != 100 for v in syn_files.values()):
        raise SystemExit("参照音 100 件・各話者 100 件がそろっていない")

    rows = []
    for variant in args.variants.split(","):
        ml = load_model(variant)
        for route in args.routes.split(","):
            label = f"{variant}/{route}"
            cache = args.out / f"emb_{variant}_{route}.npz"
            if cache.exists():
                z = np.load(cache)
                ref = z["ref"]
                syn = {s: z[s] for s in SPEAKERS}
            else:
                ref = np.stack([embed(ml, prepare(p, ml.sr, route)) for p in ref_files])
                syn = {s: np.stack([embed(ml, prepare(p, ml.sr, route)) for p in files]) for s, files in syn_files.items()}
                np.savez(cache, ref=ref, **syn)
            sets = {**syn, "Whole": np.concatenate([syn[s] for s in SPEAKERS])}
            for cond, emb in sets.items():
                fd = cf.frechet_distance(ref, emb)
                k32 = cf.kadtk_frechet_distance(ref, emb)
                rows.append({"model": variant, "route": route, "condition": cond, "n_eval": emb.shape[0],
                             "fad_float64": fd["fad"], "fad_kadtk_float32": k32, "developer": DEVELOPER[cond],
                             "mean_term": fd["mean_term"], "trace_term": fd["trace_term"]})
            print(label + "  " + "  ".join(f"{r['condition']}={r['fad_float64']:.2f}/{r['fad_kadtk_float32']:.2f}"
                                          f"(開発者 {r['developer']})" for r in rows[-7:]), flush=True)

    (args.out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
