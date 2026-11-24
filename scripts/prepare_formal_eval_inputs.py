#!/usr/bin/env python3
"""正式評価の iOS 入力を作る（口真似音を librosa.load(sr=22050) で 22050 Hz・モノラルにし、float32 の wav にする）

オリジナル PronounSE の get_spectrograms と同じ librosa.load を使う。iOS は 22050 Hz の入力をそのまま使うので、
読み込んだ波形はオリジナルと完全に一致する（48 kHz のまま渡すと AVAudioConverter の変換が合わない）。

元のデータセットは読むだけで変えない。出力は Git の対象外の場所に作る（既定 eval/fad/data/formal_inputs/）。
  <out>/<話者>/<参照>_<話者>.wav     float32、22050 Hz、モノラル
  <out>/manifest.csv                元のファイルの SHA-256、サンプル数、期待するフレーム数 T = 1 + floor(N / 256) など
既にある出力は上書きしない（台帳の元のファイルの SHA-256 が同じなら飛ばし、違えば止める）。
書いたファイルは読み直して、librosa の結果とビット単位で同じことを確かめる。

使い方
  PronounSE/venv/bin/python scripts/prepare_formal_eval_inputs.py --source ~/Desktop/Vocal_Mimicry
  （--speakers m-01,f-01 や --names <参照>_<話者>,... で一部だけ作れる）
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEAKERS = ["f-01", "m-01", "m-02", "m-03", "m-04", "m-05", "m-00"]
SR = 22050
HOP = 256
FIELDS = ["speaker", "reference", "source", "source_sha256", "source_sr", "source_samples", "output", "output_sha256",
          "samples_22050", "expected_T", "librosa", "created_at"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="Vocal_Mimicry フォルダ（話者ごとのフォルダがある）")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "eval/fad/data/formal_inputs")
    ap.add_argument("--speakers", default=",".join(SPEAKERS))
    ap.add_argument("--names", help="<参照>_<話者> をカンマ区切りで（省略時は全件）")
    args = ap.parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    if out == source or source in out.parents:
        raise SystemExit("出力はデータセットの外に置く")
    names = set(args.names.split(",")) if args.names else None

    manifest_path = out / "manifest.csv"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = {r["output"]: r for r in csv.DictReader(f)}
    new_rows = []
    made = skipped = 0
    for spk in args.speakers.split(","):
        pattern = re.compile(rf"^(?P<ref>.+)_{re.escape(spk)}\.wav$")
        files = sorted(p for p in (source / spk).iterdir() if p.is_file() and pattern.match(p.name))
        for src in files:
            ref = pattern.match(src.name)["ref"]
            if names is not None and f"{ref}_{spk}" not in names:
                continue
            rel = f"{spk}/{src.name}"
            dest = out / rel
            src_hash = sha256(src)
            if dest.exists():
                entry = manifest.get(rel)
                if entry is None or entry["source_sha256"] != src_hash or entry["output_sha256"] != sha256(dest):
                    raise SystemExit(f"{dest} が既にあり、台帳と合わない。上書きしないので止める")
                skipped += 1
                continue
            y, _ = librosa.load(src, sr=SR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".partial.wav")
            sf.write(tmp, y, SR, subtype="FLOAT")
            back, sr = sf.read(tmp, dtype="float32")
            if sr != SR or not np.array_equal(back, y):
                tmp.unlink()
                raise SystemExit(f"{rel}: 書いたファイルを読み直した値が librosa の結果と違う")
            tmp.rename(dest)
            info = sf.info(src)
            new_rows.append({
                "speaker": spk, "reference": ref, "source": str(src), "source_sha256": src_hash,
                "source_sr": info.samplerate, "source_samples": info.frames, "output": rel, "output_sha256": sha256(dest),
                "samples_22050": len(y), "expected_T": 1 + len(y) // HOP, "librosa": librosa.__version__,
                "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            })
            made += 1
    if new_rows:
        write_header = not manifest_path.exists()
        out.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    print(f"作成 {made} 件、既存のため飛ばした {skipped} 件 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
