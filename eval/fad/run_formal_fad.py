#!/usr/bin/env python3
"""正式評価（iOS 版 PronounSE）の PANNs-FAD を、オリジナル PronounSE の再計算と同じ設定で計算する

オリジナルの基準値は reproduce_original_fad.py の結果（wavegram-logmel / via22050 / 長さは切らない）を再利用する。
  - 参照音とオリジナル 600 件の埋め込み: --baseline-dir の emb_wavegram-logmel_via22050.npz
  - 丸め前の FAD: --baseline-dir の summary.json
iOS 版の埋め込みは、reproduce_original_fad.py の prepare（via22050）・embed・load_model を import してそのまま使う。
FAD は compute_fad.py の frechet_distance（float64、不偏共分散、特異値による平方根の項）。既存のコードは変えない。

確認すること（どれかに失敗すれば結果を書かずに止める）
  - 参照音 100 件、各話者 100 件・計 600 件、m-00 なし、名前の欠損・重複・余分なし、同じ波形なし
  - device/results.csv が 600 行すべて status=ok（未完成のセッションは評価しない）
  - WAV がオリジナルと同じ形式（22050 Hz・モノラル・PCM_16）
  - 参照音の埋め込みを再計算し、キャッシュと一致する（設定が同じであることの確認）
  - オリジナルの一部を再計算し、キャッシュと一致する（キャッシュの並び順の確認）
  - キャッシュから計算した FAD が summary.json の値と一致する
  - 埋め込み・FAD に NaN・Inf がない

出力（--out/<run_id>/、既にあれば止める）
  fad_results.csv / fad_results.json / evaluation_config.json / verification.json / embeddings_ios.npz

使い方
  .venv/bin/python run_formal_fad.py --run-id fp32_A_20261123 \\
      --device-dir ../../metrics/formal_eval_official/fp32_A_20261123/device --out ../../metrics/fad_official
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import compute_fad as cf
import reproduce_original_fad as ro

SPEAKERS = ro.SPEAKERS  # m-01..m-05, f-01（m-00 は学習話者なので含めない）
VARIANT, ROUTE = "wavegram-logmel", "via22050"
EXPECTED_FORMAT = (22050, 1, "PCM_16")
SPOT_CHECK_PER_SPEAKER = 3


def fail(problems: list[str]) -> None:
    if problems:
        raise SystemExit("確認に失敗したので止める:\n  " + "\n  ".join(problems))


def wav_format(p: Path) -> tuple:
    i = sf.info(p)
    return (i.samplerate, i.channels, i.subtype)


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="出力のフォルダ名（例 fp32_A_20261123）")
    ap.add_argument("--device-dir", type=Path, required=True, help="正式合成の device/（<話者>/*.wav と results.csv）")
    ap.add_argument("--dataset", type=Path, default=Path.home() / "Desktop", help="Trimmed_Explosion と PronounSE を含む")
    ap.add_argument("--baseline-dir", type=Path, default=here / "results" / "original_fad_repro_20261123")
    ap.add_argument("--out", type=Path, required=True, help="出力先の親フォルダ（例 metrics/fad_official）")
    args = ap.parse_args()

    out = args.out / args.run_id
    if out.exists():
        raise SystemExit(f"{out} は既にある。上書きしないので別の --run-id を指定する")
    torch.set_grad_enabled(False)
    problems: list[str] = []
    ver: dict = {}

    # MARK: 入力の一覧と対応
    ref_files = sorted((args.dataset / "Trimmed_Explosion").glob("*.wav"))
    ref_names = [p.stem for p in ref_files]
    orig_files = {s: sorted((args.dataset / "PronounSE" / s).glob("*.wav")) for s in SPEAKERS}
    ios_files = {s: sorted((args.device_dir / s).glob("*.wav")) for s in SPEAKERS}
    if len(ref_files) != 100:
        problems.append(f"参照音が {len(ref_files)} 件")
    extra_dirs = sorted(p.name for p in args.device_dir.iterdir() if p.is_dir() and p.name not in SPEAKERS)
    if extra_dirs:
        problems.append(f"device/ に評価対象外の話者フォルダ: {extra_dirs}")

    per_speaker = {}
    for s in SPEAKERS:
        o_keys = [p.stem.removesuffix(f"_{s}_pronounse") for p in orig_files[s]]
        i_keys = [p.stem.removesuffix(f"_{s}_pronounse_ios") for p in ios_files[s]]
        bad_names = [p.name for p in ios_files[s] if not p.stem.endswith(f"_{s}_pronounse_ios")]
        d = {"n_ios": len(ios_files[s]), "n_original": len(orig_files[s]),
             "ios_dup": len(i_keys) - len(set(i_keys)),
             "ios_missing_vs_ref": sorted(set(ref_names) - set(i_keys)),
             "ios_extra_vs_ref": sorted(set(i_keys) - set(ref_names)),
             "original_missing_vs_ref": sorted(set(ref_names) - set(o_keys)),
             "ios_equals_original_set": set(i_keys) == set(o_keys),
             "bad_names": bad_names,
             "contains_m00": any("m-00" in p.name for p in ios_files[s])}
        per_speaker[s] = d
        if (d["n_ios"] != 100 or d["ios_dup"] or d["ios_missing_vs_ref"] or d["ios_extra_vs_ref"]
                or d["original_missing_vs_ref"] or not d["ios_equals_original_set"] or bad_names or d["contains_m00"]):
            problems.append(f"{s}: 対応に問題 {d}")
    ver["sets"] = {"n_ref": len(ref_files), "per_speaker": per_speaker,
                   "n_ios_total": sum(len(v) for v in ios_files.values())}

    # MARK: 合成の記録（未完成のセッションは評価しない）
    with open(args.device_dir / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    ok_by_speaker = {s: sum(1 for r in rows if r["speaker"] == s and r["status"] == "ok") for s in SPEAKERS}
    csv_wavs = {r["output_wav"] for r in rows if r["status"] == "ok"}
    disk_wavs = {f"{s}/{p.name}" for s in SPEAKERS for p in ios_files[s]}
    nonfinite_rows = sum(1 for r in rows if r.get("wave_nonfinite", "0") not in ("0", "") or
                         r.get("postnet_nonfinite", "0") not in ("0", ""))
    ver["results_csv"] = {"rows": len(rows), "status": status_counts, "ok_by_speaker": ok_by_speaker,
                          "csv_ok_equals_disk": csv_wavs == disk_wavs, "rows_with_nonfinite": nonfinite_rows,
                          "speakers_in_csv": sorted({r["speaker"] for r in rows})}
    if len(rows) != 600 or status_counts != {"ok": 600} or csv_wavs != disk_wavs or nonfinite_rows:
        problems.append(f"results.csv が完成していない、または WAV と合わない: {ver['results_csv']}")

    # MARK: WAV の形式
    fmt_ios = {str(wav_format(p)) for v in ios_files.values() for p in v}
    fmt_orig = {str(wav_format(p)) for v in orig_files.values() for p in v}
    fmt_ref = sorted({str(wav_format(p)) for p in ref_files})
    ver["wav_format"] = {"ios": sorted(fmt_ios), "original": sorted(fmt_orig), "reference": fmt_ref,
                         "expected_synth": str(EXPECTED_FORMAT)}
    if fmt_ios != {str(EXPECTED_FORMAT)} or fmt_orig != {str(EXPECTED_FORMAT)}:
        problems.append(f"合成音の WAV の形式が想定と違う: {ver['wav_format']}")
    fail(problems)

    # MARK: 基準（オリジナル）のキャッシュ
    cache_path = args.baseline_dir / f"emb_{VARIANT}_{ROUTE}.npz"
    z = np.load(cache_path)
    base_ref = z["ref"].astype(np.float32)
    base_syn = {s: z[s].astype(np.float32) for s in SPEAKERS}
    summary = json.loads((args.baseline_dir / "summary.json").read_text())
    base_fad = {r["condition"]: r for r in summary if r["model"] == VARIANT and r["route"] == ROUTE}

    ml = ro.load_model(VARIANT)
    model_sr = ml.sr

    # 参照音を再計算してキャッシュと一致するか（同じ前処理・モデル・重みであることの確認）
    ref_emb = np.stack([ro.embed(ml, ro.prepare(p, model_sr, ROUTE)) for p in ref_files])
    ref_diff = float(np.max(np.abs(ref_emb - base_ref)))
    # オリジナルの一部を再計算（キャッシュの並びが sorted(glob) であることの確認）
    spot = {}
    for s in SPEAKERS:
        idx = np.linspace(0, len(orig_files[s]) - 1, SPOT_CHECK_PER_SPEAKER).round().astype(int)
        e = np.stack([ro.embed(ml, ro.prepare(orig_files[s][i], model_sr, ROUTE)) for i in idx])
        spot[s] = {"indices": idx.tolist(), "max_abs_diff": float(np.max(np.abs(e - base_syn[s][idx])))}
    tol = 1e-4
    ver["baseline_cache"] = {"file": str(cache_path), "ref_recomputed_max_abs_diff": ref_diff,
                             "original_spot_check": spot, "tolerance": tol}
    if ref_diff > tol or any(v["max_abs_diff"] > tol for v in spot.values()):
        problems.append(f"基準のキャッシュと再計算が一致しない: {ver['baseline_cache']}")

    # キャッシュから基準の FAD を計算し直して summary.json と一致するか
    base_sets = {**base_syn, "Whole": np.concatenate([base_syn[s] for s in SPEAKERS])}
    base_recalc = {c: cf.frechet_distance(base_ref, e)["fad"] for c, e in base_sets.items()}
    base_match = {c: abs(base_recalc[c] - base_fad[c]["fad_float64"]) for c in base_recalc}
    ver["baseline_fad_recompute_abs_diff"] = base_match
    if max(base_match.values()) > 1e-9:
        problems.append(f"基準の FAD がキャッシュから再現しない: {base_match}")
    fail(problems)

    # MARK: iOS 版の埋め込み
    ios_emb, ios_info = {}, {}
    for s in SPEAKERS:
        xs = [ro.prepare(p, model_sr, ROUTE) for p in ios_files[s]]
        hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in xs]
        if len(set(hashes)) != len(hashes):
            problems.append(f"{s}: 前処理後に同じ波形がある")
        ios_info[s] = [{"file": p.name, "samples_model_sr": int(x.shape[0]), "duration_sec": x.shape[0] / model_sr,
                        "peak": float(np.max(np.abs(x)))} for p, x in zip(ios_files[s], xs)]
        ios_emb[s] = np.stack([ro.embed(ml, x) for x in xs])
        print(f"{s}: {ios_emb[s].shape}", flush=True)
    all_emb = [ref_emb, *ios_emb.values()]
    ver["embeddings_finite"] = bool(all(np.isfinite(e).all() for e in all_emb))
    ver["embedding_shapes"] = {"ref": list(ref_emb.shape), **{s: list(e.shape) for s, e in ios_emb.items()}}
    if not ver["embeddings_finite"]:
        problems.append("埋め込みに NaN・Inf がある")
    fail(problems)

    # MARK: FAD（参照音は基準と同じキャッシュの埋め込みを使う）
    ios_sets = {**ios_emb, "Whole": np.concatenate([ios_emb[s] for s in SPEAKERS])}
    results = []
    for c in [*SPEAKERS, "Whole"]:
        fd = cf.frechet_distance(base_ref, ios_sets[c])
        b = base_fad[c]
        results.append({
            "speaker": c, "n_ref": base_ref.shape[0], "n_eval": ios_sets[c].shape[0],
            "original_fad": b["fad_float64"], "ios_fad": fd["fad"], "delta_fad": fd["fad"] - b["fad_float64"],
            "original_mean_term": b["mean_term"], "ios_mean_term": fd["mean_term"],
            "original_trace_term": b["trace_term"], "ios_trace_term": fd["trace_term"],
            "ios_fad_sqrtm_check": fd["fad_sqrtm"], "ios_sqrtm_imag_max": fd["sqrtm_imag_max"],
            "ios_fad_kadtk_float32_ref": cf.kadtk_frechet_distance(base_ref, ios_sets[c]),
            "developer_fad_reference_only": ro.DEVELOPER[c],
        })
    ver["fad_finite"] = bool(all(np.isfinite([r["ios_fad"], r["original_fad"], r["delta_fad"]]).all() for r in results))
    ver["fad_sqrtm_max_abs_diff"] = max(abs(r["ios_fad"] - r["ios_fad_sqrtm_check"]) for r in results)
    ver["whole_is_600_not_mean_of_speakers"] = results[-1]["n_eval"] == 600
    if not ver["fad_finite"]:
        problems.append("FAD に NaN・Inf がある")
    fail(problems)
    ver["problems"] = []

    # MARK: 保存
    out.mkdir(parents=True)
    with open(out / "fad_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (out / "fad_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    np.savez(out / "embeddings_ios.npz", **ios_emb,
             **{f"{s}_names": np.array([p.name for p in ios_files[s]]) for s in SPEAKERS})
    config = {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "eval_set": {"device_dir": str(args.device_dir), "speakers": SPEAKERS,
                     "excluded": ["m-00 (training speaker)"]},
        "reference": {"dir": str(args.dataset / "Trimmed_Explosion"), "n": len(ref_files),
                      "embeddings": "baseline cache (same as original FAD); recomputed and matched"},
        "baseline": {"dir": str(args.baseline_dir), "script": "eval/fad/reproduce_original_fad.py",
                     "model": VARIANT, "route": ROUTE, "original_dir": str(args.dataset / "PronounSE")},
        "method": {
            "embedding": "PANNs Wavegram_Logmel_Cnn14 (kadtk 1.1.0 PANNsModel), 2048-d, clip-level, float32",
            "checkpoint": ml.model_file.name, "checkpoint_sha256": cf.sha256_file(ml.model_file),
            "preprocess": "reproduce_original_fad.prepare(route='via22050'): soundfile float32 → mean to mono → "
                          "22050 Hz (soxr_hq, skipped if already 22050) → 32000 Hz (soxr_hq)",
            "length": "not cropped (reference, original and iOS all at full length)",
            "amplitude_normalization": None,
            "fad": "compute_fad.frechet_distance: float64, unbiased covariance (n-1), "
                   "Tr sqrt via singular values of Xc Yc^T",
            "whole": "reference 100 vs all 600 synthesized clips as one set (not mean of speakers)",
            "model_sr": model_sr, "device": "cpu",
        },
        "env": {"python": platform.python_version(), "torch": torch.__version__, "librosa": librosa.__version__,
                "numpy": np.__version__, "platform": platform.platform()},
        "ios_files": ios_info,
    }
    (out / "evaluation_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (out / "verification.json").write_text(json.dumps(ver, ensure_ascii=False, indent=2))

    print(f"\n{'Speaker':8} {'Original':>9} {'iOS':>9} {'ΔFAD':>8}")
    for r in results:
        print(f"{r['speaker']:8} {r['original_fad']:9.4f} {r['ios_fad']:9.4f} {r['delta_fad']:+8.4f}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
