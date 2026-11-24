#!/usr/bin/env python3
"""提供された評価データセット（参照音・口真似音・オリジナル PronounSE の合成音）を検証する。

データセットは読むだけで、何も書き込まない。結果は標準出力と、--out を付けたときは JSON に出す。

想定する構成（ROOT は引数で渡す）
  ROOT/Trimmed_Explosion/<参照>.wav                          参照音 100
  ROOT/Vocal_Mimicry/<話者>/<参照>_<話者>.wav                 口真似音（評価 6 話者 + 学習話者 m-00）
  ROOT/PronounSE/<話者>/<参照>_<話者>_pronounse.wav           合成音（評価 6 話者）

確かめること
  - 件数（参照 100、評価 6 話者 × 100、合成 600、m-00 の口真似 100）
  - 参照音・口真似音・合成音の名前の対応（欠損・余分・命名規則違反・話者フォルダとファイル名の話者の不一致）
  - 中身が同じファイルの重複（SHA-256）
  - WAV のメタデータ（標本化周波数・チャンネル数・長さ・形式・サブタイプ）
  - 合成音と、対応する参照音・口真似音の長さの関係（FAD の長さの扱いを決める材料）

使い方
  .venv/bin/python check_dataset.py <ROOT> [--out results/dataset_check.json]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EVAL_SPEAKERS = ["f-01", "m-01", "m-02", "m-03", "m-04", "m-05"]
TRAIN_SPEAKER = "m-00"
N_REF = 100
SPEAKER_RE = r"[fm]-\d{2}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_meta(path: Path) -> dict:
    i = sf.info(str(path))
    return {"path": str(path), "sr": i.samplerate, "channels": i.channels, "frames": i.frames,
            "duration_sec": i.frames / i.samplerate, "format": i.format, "subtype": i.subtype,
            "sha256": sha256_file(path)}


def summarize(metas: list[dict]) -> dict:
    d = np.array([m["duration_sec"] for m in metas])
    return {
        "n": len(metas),
        "sr": dict(collections.Counter(m["sr"] for m in metas)),
        "channels": dict(collections.Counter(m["channels"] for m in metas)),
        "format/subtype": dict(collections.Counter(f"{m['format']}/{m['subtype']}" for m in metas)),
        "duration_sec": {"min": float(d.min()), "median": float(np.median(d)), "max": float(d.max()),
                         "mean": float(d.mean())} if len(d) else {},
    }


def list_files(folder: Path) -> tuple[list[Path], list[Path]]:
    """(wav, wav 以外) を返す。隠しファイル（.DS_Store など）は wav 以外に入れる"""
    files = sorted(p for p in folder.iterdir() if p.is_file())
    return [p for p in files if p.suffix == ".wav" and not p.name.startswith(".")], \
           [p for p in files if not (p.suffix == ".wav" and not p.name.startswith("."))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, help="結果の JSON（データセットの外に書く）")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.out and root in args.out.resolve().parents:
        raise SystemExit("--out はデータセットの外に置く")

    problems: list[str] = []
    report: dict = {"root": str(root)}

    # MARK: 参照音
    ref_dir = root / "Trimmed_Explosion"
    ref_wavs, ref_other = list_files(ref_dir)
    subdirs = [p.name for p in ref_dir.iterdir() if p.is_dir()]
    if subdirs:
        problems.append(f"Trimmed_Explosion にサブフォルダがある: {subdirs}")
    if ref_other:
        problems.append(f"Trimmed_Explosion に wav 以外のファイル: {[p.name for p in ref_other]}")
    ref_names = {p.stem for p in ref_wavs}
    if len(ref_wavs) != N_REF:
        problems.append(f"参照音が {len(ref_wavs)} 件（{N_REF} 件を想定）")
    bad_ref = [n for n in ref_names if re.search(rf"_{SPEAKER_RE}$", n) or n.endswith("_pronounse")]
    if bad_ref:
        problems.append(f"参照音の名前に話者 ID や _pronounse が付いている: {bad_ref}")
    ref_meta = {p.stem: wav_meta(p) for p in ref_wavs}

    # MARK: 口真似音・合成音
    def check_speaker_dirs(kind: str, folder: Path, expected: list[str], suffix: str) -> dict[str, dict[str, dict]]:
        found = sorted(p.name for p in folder.iterdir() if p.is_dir())
        stray = [p.name for p in folder.iterdir() if p.is_file()]
        if stray:
            problems.append(f"{kind} の直下にファイル: {stray}")
        if found != sorted(expected):
            problems.append(f"{kind} の話者フォルダ {found}（想定 {sorted(expected)}）")
        out: dict[str, dict[str, dict]] = {}
        for spk in found:
            wavs, other = list_files(folder / spk)
            if other:
                problems.append(f"{kind}/{spk} に wav 以外のファイル: {[p.name for p in other]}")
            if any(p.is_dir() for p in (folder / spk).iterdir()):
                problems.append(f"{kind}/{spk} にサブフォルダがある")
            pat = re.compile(rf"^(?P<ref>.+)_(?P<spk>{SPEAKER_RE}){re.escape(suffix)}$")
            by_ref: dict[str, dict] = {}
            for p in wavs:
                m = pat.match(p.stem)
                if not m:
                    problems.append(f"{kind}/{spk}/{p.name}: 命名規則に合わない")
                    continue
                if m["spk"] != spk:
                    problems.append(f"{kind}/{spk}/{p.name}: ファイル名の話者 {m['spk']} がフォルダと違う")
                if m["ref"] in by_ref:
                    problems.append(f"{kind}/{spk}: 参照 {m['ref']} が重複")
                by_ref[m["ref"]] = wav_meta(p)
            missing, extra = ref_names - by_ref.keys(), by_ref.keys() - ref_names
            if missing:
                problems.append(f"{kind}/{spk}: 参照音に対応するファイルがない {len(missing)} 件: {sorted(missing)[:5]}")
            if extra:
                problems.append(f"{kind}/{spk}: 参照音にないファイル {len(extra)} 件: {sorted(extra)[:5]}")
            if len(wavs) != N_REF:
                problems.append(f"{kind}/{spk}: {len(wavs)} 件（{N_REF} 件を想定）")
            out[spk] = by_ref
        return out

    mim = check_speaker_dirs("Vocal_Mimicry", root / "Vocal_Mimicry", EVAL_SPEAKERS + [TRAIN_SPEAKER], "")
    syn = check_speaker_dirs("PronounSE", root / "PronounSE", EVAL_SPEAKERS, "_pronounse")

    # 合成音には口真似音がそろっているか（合成音 ↔ 口真似音の対応）
    for spk, by_ref in syn.items():
        lack = by_ref.keys() - mim.get(spk, {}).keys()
        if lack:
            problems.append(f"PronounSE/{spk}: 対応する口真似音がない {sorted(lack)[:5]}")

    # MARK: 内容の重複（SHA-256）
    all_meta = [("ref", n, m) for n, m in ref_meta.items()]
    all_meta += [(f"mimicry/{s}", n, m) for s, d in mim.items() for n, m in d.items()]
    all_meta += [(f"pronounse/{s}", n, m) for s, d in syn.items() for n, m in d.items()]
    by_hash = collections.defaultdict(list)
    for kind, n, m in all_meta:
        by_hash[m["sha256"]].append(f"{kind}/{n}")
    dups = [v for v in by_hash.values() if len(v) > 1]
    if dups:
        problems.append(f"中身が同じファイル {len(dups)} 組: {dups[:5]}")

    # MARK: 長さの関係（合成音 ÷ 参照音、合成音 ÷ 口真似音）
    ratios_ref, ratios_mim, diff_mim = [], [], []
    for spk, by_ref in syn.items():
        for r, m in by_ref.items():
            if r in ref_meta:
                ratios_ref.append(m["duration_sec"] / ref_meta[r]["duration_sec"])
            if r in mim.get(spk, {}):
                ratios_mim.append(m["duration_sec"] / mim[spk][r]["duration_sec"])
                diff_mim.append(m["duration_sec"] - mim[spk][r]["duration_sec"])

    def stats(a):
        a = np.array(a)
        return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())} if len(a) else {}

    report["reference"] = summarize(list(ref_meta.values()))
    report["mimicry"] = {s: summarize(list(d.values())) for s, d in mim.items()}
    report["pronounse"] = {s: summarize(list(d.values())) for s, d in syn.items()}
    report["pronounse_all"] = summarize([m for d in syn.values() for m in d.values()])
    report["mimicry_eval_all"] = summarize([m for s, d in mim.items() if s in EVAL_SPEAKERS for m in d.values()])
    report["length_relation"] = {"pronounse_over_reference": stats(ratios_ref),
                                 "pronounse_over_mimicry": stats(ratios_mim),
                                 "pronounse_minus_mimicry_sec": stats(diff_mim)}
    report["rtf_m00_mimicry"] = len(mim.get(TRAIN_SPEAKER, {}))
    report["problems"] = problems

    # MARK: 出力
    def line(label, s):
        dur = s["duration_sec"]
        print(f"  {label:<14} n={s['n']:<4} sr={s['sr']} ch={s['channels']} {s['format/subtype']} "
              f"長さ {dur.get('min', 0):.2f}/{dur.get('median', 0):.2f}/{dur.get('max', 0):.2f} 秒（最小/中央/最大）")

    print(f"データセット: {root}")
    line("参照音", report["reference"])
    for s in sorted(mim):
        line(f"口真似 {s}", report["mimicry"][s])
    for s in sorted(syn):
        line(f"合成 {s}", report["pronounse"][s])
    line("合成 全体", report["pronounse_all"])
    lr = report["length_relation"]
    print("  長さの比 合成/参照  ", {k: round(v, 3) for k, v in lr["pronounse_over_reference"].items()})
    print("  長さの比 合成/口真似", {k: round(v, 3) for k, v in lr["pronounse_over_mimicry"].items()})
    print("  長さの差 合成-口真似（秒）", {k: round(v, 3) for k, v in lr["pronounse_minus_mimicry_sec"].items()})
    print(f"  RTF 用 m-00 の口真似音: {report['rtf_m00_mimicry']} 件")
    print(f"問題: {len(problems)} 件")
    for p in problems:
        print(f"  - {p}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"→ {args.out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
