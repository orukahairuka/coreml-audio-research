#!/usr/bin/env python3
"""正式評価の一括合成の結果（端末から取り出したセッション）を全件確かめる

確かめること
  1. 話者ごとの件数（--per-speaker 件）と合計
  2. WAV（PCM16）と NPY（float32）が一対一で対応し、余分なファイルや書きかけのファイルが無い
  3. 出力の名前（<参照>_<話者>_pronounse_ios）が入力と対応し、参照音（Trimmed_Explosion/<参照>.wav）がある
  4. WAV が 22050 Hz・モノラル・PCM_16
  5. 出力のサンプル数が T × 256（T は入力の台帳の expected_T と results.csv の両方）
  6. NaN・Inf が無い（NPY で数える）
  7. 欠損・重複が無い。対象の入力がすべて results.csv に status=ok で 1 回だけある
  8. 元の口真似音が、入力を作ったときから変わっていない（台帳の source_sha256 と比べる）
振幅が ±1 を超えた件数とサンプル数も数える（PyTorch 版も超えることがあるので、それだけで異常とはしない）。

使い方
  PronounSE/venv/bin/python scripts/verify_formal_session.py --session metrics/formal_eval_official/<session>/device \
      --source ~/Desktop/Vocal_Mimicry --reference-dir ~/Desktop/Trimmed_Explosion
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEAKERS = ["f-01", "m-01", "m-02", "m-03", "m-04", "m-05"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, default=REPO_ROOT / "eval/fad/data/formal_inputs")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--speakers", default=",".join(SPEAKERS))
    ap.add_argument("--per-speaker", type=int, default=100)
    args = ap.parse_args()
    speakers = args.speakers.split(",")
    problems: list[str] = []

    manifest = {r["output"]: r for r in csv.DictReader(open(args.inputs / "manifest.csv"))}
    targets = {k: v for k, v in manifest.items() if v["speaker"] in speakers}
    rows = list(csv.DictReader(open(args.session / "results.csv")))
    ok = [r for r in rows if r["status"] == "ok"]
    ok_by_input = collections.Counter(r["input_file"] for r in ok)

    # 7. 欠損・重複・記録
    missing = sorted(set(targets) - set(ok_by_input))
    dup = sorted(k for k, v in ok_by_input.items() if v > 1)
    extra = sorted(set(ok_by_input) - set(targets))
    if missing:
        problems.append(f"status=ok の記録が無い入力 {len(missing)} 件: {missing[:5]}")
    if dup:
        problems.append(f"status=ok が重複している入力 {len(dup)} 件: {dup[:5]}")
    if extra:
        problems.append(f"対象外の入力の ok 行 {len(extra)} 件: {extra[:5]}")
    other_status = collections.Counter(r["status"] for r in rows if r["status"] != "ok")

    # 1. 件数
    per_speaker = collections.Counter(r["speaker"] for r in ok if r["input_file"] in targets)
    for spk in speakers:
        if per_speaker[spk] != args.per_speaker:
            problems.append(f"{spk}: {per_speaker[spk]} 件（{args.per_speaker} 件を想定）")

    # 2. ファイルの一対一
    wavs = {p.relative_to(args.session).as_posix() for p in args.session.glob("*/*.wav")}
    npys = {p.relative_to(args.session).as_posix() for p in args.session.glob("*/*.npy")}
    partial = [p for p in wavs | npys if ".partial." in p]
    if partial:
        problems.append(f"書きかけのファイル {len(partial)} 件: {partial[:5]}")
    expected_wav = {r["output_wav"] for r in ok}
    expected_npy = {r["output_f32"] for r in ok}
    if wavs - expected_wav or expected_wav - wavs:
        problems.append(f"WAV と results.csv が合わない（余分 {sorted(wavs - expected_wav)[:3]}、無い {sorted(expected_wav - wavs)[:3]}）")
    if npys - expected_npy or expected_npy - npys:
        problems.append(f"NPY と results.csv が合わない（余分 {sorted(npys - expected_npy)[:3]}、無い {sorted(expected_npy - npys)[:3]}）")

    clip_files = clip_samples = nonfinite = 0
    peaks = []
    for r in ok:
        m = targets.get(r["input_file"])
        stem = f"{r['reference']}_{r['speaker']}"
        # 3. 名前と参照音
        if Path(r["input_file"]).stem != stem or Path(r["output_wav"]).name != f"{stem}_pronounse_ios.wav" \
                or Path(r["output_f32"]).name != f"{stem}_pronounse_ios_f32.npy":
            problems.append(f"{r['input_file']}: 入力と出力の名前が対応しない")
        if not (args.reference_dir / f"{r['reference']}.wav").exists():
            problems.append(f"{r['input_file']}: 参照音 {r['reference']}.wav が無い")
        wav_path, npy_path = args.session / r["output_wav"], args.session / r["output_f32"]
        if not wav_path.exists() or not npy_path.exists():
            continue
        # 4. 形式
        info = sf.info(wav_path)
        if (info.samplerate, info.channels, info.subtype, info.format) != (22050, 1, "PCM_16", "WAV"):
            problems.append(f"{r['output_wav']}: {info.samplerate} Hz, {info.channels} ch, {info.format}/{info.subtype}")
        # 5. 長さ
        T = int(r["T"])
        x = np.load(npy_path).ravel()
        if m is not None and int(m["expected_T"]) != T:
            problems.append(f"{r['input_file']}: T={T} が台帳の {m['expected_T']} と違う")
        if info.frames != T * 256 or x.size != T * 256 or int(r["output_samples"]) != T * 256:
            problems.append(f"{r['output_wav']}: 長さ（wav {info.frames}、npy {x.size}）が T×256={T * 256} と違う")
        # 6. NaN・Inf と振幅
        nf = int((~np.isfinite(x)).sum())
        nonfinite += nf
        if nf:
            problems.append(f"{r['output_f32']}: NaN/Inf {nf} 個")
        finite = x[np.isfinite(x)]
        peaks.append(float(np.abs(finite).max()) if finite.size else float("nan"))
        c = int((np.abs(finite) >= 1).sum())
        clip_samples += c
        clip_files += c > 0
        # WAV は NPY を ±1 で切って量子化したもの（丸めの違いは 1 LSB まで）
        pcm, _ = sf.read(wav_path, dtype="int16")
        if pcm.size == x.size and np.abs(pcm.astype(np.float64) - np.clip(x, -1, 1) * 32768).max() > 1.5:
            problems.append(f"{r['output_wav']}: WAV と NPY が対応しない")

    # 8. 元の口真似音が変わっていない
    changed = [k for k, v in targets.items() if sha256(args.source / v["speaker"] / Path(k).name) != v["source_sha256"]]
    if changed:
        problems.append(f"元の口真似音が台帳と違う {len(changed)} 件: {changed[:5]}")

    report = {
        "session": str(args.session), "targets": len(targets), "ok": len(ok), "per_speaker": dict(per_speaker),
        "other_status": dict(other_status), "wav_files": len(wavs), "npy_files": len(npys),
        "nonfinite_values": nonfinite, "files_with_abs_ge_1": clip_files, "samples_with_abs_ge_1": clip_samples,
        "peak": {"min": min(peaks), "median": float(np.median(peaks)), "max": max(peaks)} if peaks else None,
        "source_unchanged": not changed, "problems": problems,
    }
    (args.session.parent / "verify_full.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
