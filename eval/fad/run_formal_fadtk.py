#!/usr/bin/env python3
"""正式評価の FAD を DCASE2024 Task7 版 fadtk（commit 9c989db、panns-wavegram-logmel）で計算する

fadtk の CLI（fadtk/__main__.py）が 1 つのモデルについて行う処理を、公開 API で同じ順に呼ぶ:
  1. fadtk.fad_batch.cache_embedding_files(<dir>, ml, workers)   … 音声の読み込み・リサンプリング・埋め込み
  2. fadtk.FrechetAudioDistance(ml, load_model=False).score(<ref dir>, <eval dir>)  … 統計と FAD
CLI を直接使わないのは、CLI が起動時に get_all_models() で CLAP・MERT・Whisper など全モデルを作り、
それらの依存（数 GB）がないと動かないため。

fadtk の処理には手を入れない。この環境で動かすために次の 2 点だけを外から差し替える（値は変わらない）:
  - torch.load に map_location="cpu" を付ける（重みが CUDA 上で保存されており、CPU だけの Mac では読めない）
  - PANNsModel.load_model が使わない Cnn14 32k/16k の重みまでダウンロードするのを止める
    （読み込むのは Wavegram_Logmel_Cnn14_mAP=0.439.pth だけ。このファイルが無ければ止める）

fadtk は音声のフォルダの中に convert/・embeddings/・stats/ を書くので、元の音声は --work-dir にコピーしてから使う。
作業フォルダは新しく作る（既にあれば止める。古い stats/ のキャッシュを使わないため）。
  <work>/ref/                      参照音 100 件（通常版・iOS 版で共通）
  <work>/<set>/<話者>/              合成音 100 件ずつ
  <work>/<set>/whole/              合成音 600 件をまとめたもの（Whole）

使い方（fadtk の venv で実行。README の「fadtk の環境」を参照）
  <fadtk venv>/bin/python run_formal_fadtk.py --run-id fp32_A_20261123 \\
      --eval ios_fp32=../../metrics/formal_eval_official/fp32_A_20261123/device \\
      --work-dir <作業フォルダ> --out ../../metrics/fad_official_fadtk
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import fadtk
import fadtk.model_loader as fml
from fadtk.fad import FrechetAudioDistance
from fadtk.fad import log as fadtk_log
from fadtk.fad_batch import cache_embedding_files
from fadtk.utils import get_cache_embedding_path

MODEL_NAME = "panns-wavegram-logmel"
CKPT_NAME = "Wavegram_Logmel_Cnn14_mAP=0.439.pth"
CKPT_URL = "https://zenodo.org/records/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth"  # fadtk が取得する URL
FADTK_COMMIT = "9c989dba2ac74700ec58c775e51d49539b40dfbf"
SPEAKERS = ["m-01", "m-02", "m-03", "m-04", "m-05", "f-01"]  # m-00（学習話者）は含めない
CKPT_PATH = Path(fml.__file__).parent / "panns" / "ckpt" / CKPT_NAME


# MARK: - 環境に合わせた差し替え（spawn の子プロセスでも効くようにモジュールの読み込み時に行う）

def _skip_download(url, path):
    if Path(path).name == CKPT_NAME:
        raise SystemExit(f"{CKPT_NAME} がない: {path}")
    print(f"skip download (unused by {MODEL_NAME}): {Path(path).name}", file=sys.stderr)


_orig_torch_load = torch.load
fml.download_file = _skip_download
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "map_location": "cpu"})


# MARK: - 入力の確認

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(src_root: Path, suffix: str) -> dict[str, list[Path]]:
    """<src_root>/<話者>/<参照音名>_<話者>_<suffix>.wav を話者ごとに集める"""
    return {s: sorted((src_root / s).glob(f"*_{s}_{suffix}.wav")) for s in SPEAKERS}


def check_set(label: str, files: dict[str, list[Path]], ref_names: list[str], suffix: str, problems: list[str]) -> dict:
    out = {}
    for s in SPEAKERS:
        keys = [p.name.removesuffix(f"_{s}_{suffix}.wav") for p in files[s]]
        all_wavs = sorted(p.name for p in files[s][0].parent.glob("*.wav")) if files[s] else []
        d = {"n": len(files[s]), "dup": len(keys) - len(set(keys)),
             "missing_vs_ref": sorted(set(ref_names) - set(keys)), "extra_vs_ref": sorted(set(keys) - set(ref_names)),
             "other_wavs_in_folder": sorted(set(all_wavs) - {p.name for p in files[s]}),
             "contains_m00": any("m-00" in p.name for p in files[s]),
             "formats": sorted({str((sf.info(p).samplerate, sf.info(p).channels, sf.info(p).subtype)) for p in files[s]})}
        out[s] = d
        if d["n"] != 100 or d["dup"] or d["missing_vs_ref"] or d["extra_vs_ref"] or d["other_wavs_in_folder"] \
                or d["contains_m00"]:
            problems.append(f"{label}/{s}: {d}")
    return out


def check_device_results(device_dir: Path, files: dict[str, list[Path]], problems: list[str]) -> dict:
    """正式合成の results.csv が 600 行すべて ok で、WAV と一致するか（未完成のセッションは評価しない）"""
    csv_path = device_dir / "results.csv"
    if not csv_path.exists():
        problems.append(f"{csv_path} がない")
        return {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    status: dict[str, int] = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    csv_ok = {r["output_wav"] for r in rows if r["status"] == "ok"}
    disk = {f"{s}/{p.name}" for s in SPEAKERS for p in files[s]}
    d = {"rows": len(rows), "status": status, "csv_ok_equals_disk": csv_ok == disk,
         "speakers_in_csv": sorted({r["speaker"] for r in rows})}
    if len(rows) != 600 or status != {"ok": 600} or csv_ok != disk:
        problems.append(f"{csv_path}: 正式合成が完成していない、または WAV と合わない {d}")
    return d


def copy_files(files: list[Path], dst: Path) -> None:
    dst.mkdir(parents=True)
    for p in files:
        shutil.copy2(p, dst / p.name)


# MARK: - main

def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="出力のフォルダ名（例 fp32_A_20261123）")
    ap.add_argument("--eval", required=True, metavar="NAME=DEVICE_DIR",
                    help="iOS 版の正式合成の device/（<話者>/*_pronounse_ios.wav と results.csv）")
    ap.add_argument("--dataset", type=Path, default=Path.home() / "Desktop",
                    help="Trimmed_Explosion/（参照音）と PronounSE/（通常版）を含むフォルダ")
    ap.add_argument("--work-dir", type=Path, required=True, help="fadtk が書き込む作業フォルダ（新しく作る）")
    ap.add_argument("--out", type=Path, default=here.parent.parent / "metrics" / "fad_official_fadtk")
    ap.add_argument("--workers", type=int, default=1, help="cache_embedding_files の workers")
    ap.add_argument("--compare-custom", type=Path, help="独自実装の fad_results.csv（比較用。任意）")
    ap.add_argument("--compare-trial", type=Path, help="以前の fadtk 試算の summary.json（整合の確認用。任意）")
    args = ap.parse_args()

    eval_name, _, eval_dir = args.eval.partition("=")
    if not eval_dir or eval_name in ("ref", "original"):
        raise SystemExit("--eval は NAME=DIR（NAME は ref・original 以外）")
    eval_dir = Path(eval_dir).resolve()
    out = args.out / args.run_id
    if out.exists():
        raise SystemExit(f"{out} は既にある。上書きしないので別の --run-id を指定する")
    if args.work_dir.exists():
        raise SystemExit(f"{args.work_dir} は既にある。fadtk のキャッシュを使い回さないよう新しいフォルダを指定する")
    if not CKPT_PATH.exists():
        raise SystemExit(f"{CKPT_PATH} がない（README の「fadtk の環境」を参照）")

    problems: list[str] = []
    ver: dict = {}

    # 入力の確認
    ref_files = sorted((args.dataset / "Trimmed_Explosion").glob("*.wav"))
    ref_names = [p.stem for p in ref_files]
    if len(ref_files) != 100 or len(set(ref_names)) != 100:
        problems.append(f"参照音が {len(ref_files)} 件")
    sets = {"original": (collect(args.dataset / "PronounSE", "pronounse"), "pronounse"),
            eval_name: (collect(eval_dir, "pronounse_ios"), "pronounse_ios")}
    ver["inputs"] = {"ref": {"dir": str(args.dataset / "Trimmed_Explosion"), "n": len(ref_files),
                             "formats": sorted({str((sf.info(p).samplerate, sf.info(p).channels, sf.info(p).subtype))
                                                for p in ref_files})}}
    for label, (files, suffix) in sets.items():
        ver["inputs"][label] = check_set(label, files, ref_names, suffix, problems)
    ver["inputs"][eval_name + "_results_csv"] = check_device_results(eval_dir, sets[eval_name][0], problems)
    if problems:
        raise SystemExit("入力の確認に失敗したので止める:\n  " + "\n  ".join(problems))
    sources = ref_files + [p for files, _ in sets.values() for v in files.values() for p in v]
    src_hash_before = {str(p): sha256_file(p) for p in sources}

    # 作業フォルダにコピー（元の音声には何も書かない）
    W = args.work_dir.resolve()
    copy_files(ref_files, W / "ref")
    eval_dirs: dict[str, dict[str, Path]] = {}
    for label, (files, _) in sets.items():
        eval_dirs[label] = {}
        for s in SPEAKERS:
            copy_files(files[s], W / label / s)
            eval_dirs[label][s] = W / label / s
        copy_files([p for s in SPEAKERS for p in files[s]], W / label / "whole")
        eval_dirs[label]["Whole"] = W / label / "whole"
    copied = {str(p): sha256_file(p) for p in W.rglob("*.wav")}
    ver["copies_identical_to_sources"] = sorted(copied.values()) == sorted(
        [src_hash_before[str(p)] for p in ref_files] +
        [src_hash_before[str(p)] for files, _ in sets.values() for v in files.values() for p in v] * 2)

    # fadtk の警告を記録する
    warnings: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                warnings.append(record.getMessage())
    fadtk_log.addHandler(_Collect())

    # 1. 埋め込み（fadtk の CLI と同じ cache_embedding_files）
    ml = fml.PANNsModel("wavegram-logmel")
    assert ml.name == MODEL_NAME
    all_dirs = [W / "ref"] + [d for v in eval_dirs.values() for d in v.values()]
    for d in all_dirs:
        cache_embedding_files(d, ml, workers=args.workers)
        print(f"embedded: {d.relative_to(W)}", flush=True)

    # 2. FAD（fadtk の CLI と同じ FrechetAudioDistance.score）
    fad = FrechetAudioDistance(ml, audio_load_worker=args.workers, load_model=False)
    scores: dict[str, dict[str, float]] = {}
    for label, dirs in eval_dirs.items():
        scores[label] = {c: float(fad.score(str(W / "ref"), str(d))) for c, d in dirs.items()}
        print(label, {c: round(v, 4) for c, v in scores[label].items()}, flush=True)

    # 埋め込みの確認（件数・NaN/Inf・Whole が話者別の和集合と同じ）
    def emb_of(d: Path) -> dict[str, np.ndarray]:
        return {p.name: np.load(get_cache_embedding_path(ml.name, p)) for p in sorted(d.glob("*.wav"))}
    ref_emb = emb_of(W / "ref")
    emb_check = {"ref": {"n": len(ref_emb), "shape": list(next(iter(ref_emb.values())).shape),
                         "dtype": str(next(iter(ref_emb.values())).dtype),
                         "finite": bool(all(np.isfinite(e).all() for e in ref_emb.values()))}}
    saved = {"ref": np.concatenate(list(ref_emb.values())), "ref_names": np.array(list(ref_emb))}
    for label, dirs in eval_dirs.items():
        per = {s: emb_of(dirs[s]) for s in SPEAKERS}
        whole = emb_of(dirs["Whole"])
        union = {k: v for s in SPEAKERS for k, v in per[s].items()}
        emb_check[label] = {
            "n_per_speaker": {s: len(per[s]) for s in SPEAKERS}, "n_whole": len(whole),
            "finite": bool(all(np.isfinite(e).all() for e in [*union.values(), *whole.values()])),
            "whole_names_equal_union": set(whole) == set(union),
            "whole_embeddings_equal_union_max_abs_diff":
                float(max(np.max(np.abs(whole[k].astype(np.float64) - union[k].astype(np.float64))) for k in whole)),
        }
        for s in SPEAKERS:
            saved[f"{label}__{s}"] = np.concatenate(list(per[s].values()))
            saved[f"{label}__{s}__names"] = np.array(list(per[s]))
    ver["embeddings"] = emb_check

    # stats/ のキャッシュの件数が入力と同じか（fadtk は embeddings/ の npy をすべて読む）
    ver["fadtk_embedding_files"] = {str(d.relative_to(W)): len(list((d / "embeddings" / ml.name).glob("*.npy")))
                                    for d in all_dirs}
    expect = {"ref": 100, **{f"{l}/{s}": 100 for l in eval_dirs for s in SPEAKERS},
              **{f"{l}/whole": 600 for l in eval_dirs}}
    if ver["fadtk_embedding_files"] != expect:
        problems.append(f"embeddings/ の件数が想定と違う: {ver['fadtk_embedding_files']}")
    if not all(v.get("finite", True) for v in emb_check.values()):
        problems.append("埋め込みに NaN・Inf がある")
    for label in eval_dirs:
        e = emb_check[label]
        if e["n_whole"] != 600 or not e["whole_names_equal_union"] or e["whole_embeddings_equal_union_max_abs_diff"] != 0:
            problems.append(f"{label}: Whole が話者別の 600 件と一致しない")
    if not all(np.isfinite(v) for sc in scores.values() for v in sc.values()):
        problems.append("FAD に NaN・Inf がある")
    if not ver["copies_identical_to_sources"]:
        problems.append("コピーが元の音声と一致しない")
    ver["sources_unchanged"] = {str(p): sha256_file(p) for p in sources} == src_hash_before
    if not ver["sources_unchanged"]:
        problems.append("元の音声が変わった")
    ver["fadtk_warnings"] = warnings
    ver["problems"] = problems

    # 結果
    conds = [*SPEAKERS, "Whole"]
    results = [{"speaker": c, "n_ref": 100, "n_eval": 600 if c == "Whole" else 100,
                "original_fad": scores["original"][c], f"{eval_name}_fad": scores[eval_name][c],
                "delta_fad": scores[eval_name][c] - scores["original"][c]} for c in conds]

    # 以前の結果との比較（任意）
    if args.compare_trial and args.compare_trial.exists():
        trial = {r["condition"]: r["fadtk"] for r in json.loads(args.compare_trial.read_text())["rows"]}
        ver["original_vs_previous_fadtk_trial_abs_diff"] = {c: abs(scores["original"][c] - trial[c]) for c in conds}
    if args.compare_custom and args.compare_custom.exists():
        with open(args.compare_custom, newline="") as f:
            custom = {r["speaker"]: r for r in csv.DictReader(f)}
        for r in results:
            c = custom[r["speaker"]]
            r.update({"custom_original_fad": float(c["original_fad"]), "custom_ios_fad": float(c["ios_fad"]),
                      "custom_delta_fad": float(c["delta_fad"])})

    # 保存
    out.mkdir(parents=True)
    with open(out / "fad_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (out / "fad_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    np.savez(out / "embeddings_fadtk.npz", **saved)
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True).stdout
    (out / "fadtk_venv_freeze.txt").write_text(freeze)
    config = {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "command": " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
        "fadtk": {"repo": "https://github.com/DCASE2024-Task7-Sound-Scene-Synthesis/fadtk",
                  "commit": FADTK_COMMIT, "package_version": "1.0.0", "installed_at": str(Path(fadtk.__file__).parent),
                  "api_calls": ["fadtk.fad_batch.cache_embedding_files(dir, ml, workers)",
                                "fadtk.FrechetAudioDistance(ml, load_model=False).score(ref_dir, eval_dir)"],
                  "standard_processing": {
                      "load_resample": "FrechetAudioDistance.load_audio: TORCHAUDIO_RESAMPLING=True → torchaudio.load、"
                                       "チャンネル平均、torchaudio Resample(sinc_interp_kaiser, width 64, rolloff 0.9476, "
                                       "beta 14.77) で 32 kHz へ、PCM_S16 で保存 → int16/32768 で読み込み",
                      "embedding": "PANNsModel('wavegram-logmel')._get_embedding: Wavegram_Logmel_Cnn14 の 'embedding'"
                                   "（2048 次元）、float16 で保存",
                      "stats": "calculate_embd_statistics_online: 平均と不偏共分散（n−1）",
                      "fad": "calc_frechet_distance: 固有値分解による (Σ1Σ2)^{1/2}、float64"},
                  "environment_patches": [
                      "torch.load に map_location='cpu'（重みが CUDA 上で保存されており CPU だけでは読めない。値は変わらない）",
                      "使わない Cnn14 32k/16k の重みのダウンロードを止める（読み込むのは Wavegram-Logmel のみ）"],
                  "cli_not_used_reason": "CLI は起動時に get_all_models() で全モデルを作り、CLAP・MERT 等の依存が必要になる"},
        "model": {"name": MODEL_NAME, "checkpoint": CKPT_NAME, "checkpoint_url": CKPT_URL,
                  "checkpoint_sha256": sha256_file(CKPT_PATH), "sr": ml.sr, "num_features": ml.num_features,
                  "device": str(ml.device)},
        "data": {"reference": str(args.dataset / "Trimmed_Explosion"), "original": str(args.dataset / "PronounSE"),
                 eval_name: str(eval_dir), "speakers": SPEAKERS, "excluded": ["m-00 (training speaker)"],
                 "whole": "reference 100 vs 600 synthesized clips in one folder (not mean of speakers)",
                 "work_dir": str(W)},
        "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "platform": platform.platform()},
    }
    (out / "evaluation_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (out / "verification.json").write_text(json.dumps(ver, ensure_ascii=False, indent=2))

    print(f"\n{'Speaker':8} {'Original':>9} {eval_name:>9} {'ΔFAD':>8}")
    for r in results:
        print(f"{r['speaker']:8} {r['original_fad']:9.4f} {r[f'{eval_name}_fad']:9.4f} {r['delta_fad']:+8.4f}")
    print(f"→ {out}")
    if problems:
        print("確認で問題が見つかった（verification.json の problems）:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
