#!/usr/bin/env python3
"""PANNs の埋め込みによる FAD（Fréchet Audio Distance）を計算する。

iOS アプリと推論コードからは独立した評価環境（eval/fad/.venv）で動かす。

方法
  - 埋め込み: PANNs Wavegram-Logmel-CNN14（AudioSet で学習、32 kHz、2048 次元、クリップごとに 1 つ）。
    DCASE 2024 Task 7（Sound Scene Synthesis）の公式評価と同じモデル。モデルの定義と重みの取得は
    kadtk（YoonjinXD/kadtk、DCASE 2024 Task 7 の fadtk の派生）の PANNsModel をそのまま使う。
  - FAD: 2 つの集合の埋め込みの平均 μ と共分散 Σ（不偏、n−1 で割る）から
        ||μ_r − μ_e||² + Tr(Σ_r) + Tr(Σ_e) − 2 Tr((Σ_r Σ_e)^{1/2})
    を float64 で求める。最後の項は sqrtm を使わずに特異値の和で求める（frechet_distance の説明）。

前処理（入力のフォルダには何も書かない）
  1. wav を float で読み、複数チャンネルなら平均してモノラルにする
  2. --common-sr（既定 22050 Hz）に変換して帯域をそろえる
  3. 長さをそろえる（--align）
       shortest-by-name（既定）: 同じファイル名（拡張子を除く）の音を、すべての集合の中で最も短い長さに、
                                 先頭から切る。どの集合にも同じ名前の組がそろっていなければ止める
       fixed: すべての音を --crop-sec 秒に先頭から切る
       none: 切らない
     0 埋めで伸ばすことはしない
  4. 32 kHz に変換して PANNs に入れる。振幅の正規化はしない
  変換は soxr（librosa の soxr_hq）で行う。

標本の独立性
  同じ集合の中に、前処理後の波形が同じ音（同じ入力の推論を繰り返した出力など）があれば止める。
  1 つの入力からは 1 つの音だけを集合に入れる。

使い方
  .venv/bin/python compute_fad.py --ref <参照音のフォルダ> \\
      --eval nvidia=<フォルダ> --eval iphone13=<フォルダ> --eval iphone17pro=<フォルダ> \\
      --out results/<名前>.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import scipy.linalg
import soundfile as sf
import torch

PANNS_SR = 32000
MODEL_VARIANT = "wavegram-logmel"


# MARK: - 音声の読み込みと前処理

@dataclass
class Clip:
    name: str
    x: np.ndarray            # common_sr のモノラル float32
    info: dict = field(default_factory=dict)


def list_wavs(folder: Path) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".wav", ".flac"))
    if not files:
        raise SystemExit(f"音声ファイルがない: {folder}")
    return files


def load_clip(path: Path, common_sr: int) -> Clip:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    info = {"file": path.name, "orig_sr": sr, "orig_channels": audio.shape[1], "orig_samples": audio.shape[0],
            "orig_duration_sec": audio.shape[0] / sr}
    x = audio.mean(axis=1)
    if sr != common_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=common_sr, res_type="soxr_hq")
    return Clip(path.stem, x.astype(np.float32), info)


def load_set(folder: Path, common_sr: int) -> list[Clip]:
    clips = [load_clip(p, common_sr) for p in list_wavs(folder)]
    names = [c.name for c in clips]
    if len(set(names)) != len(names):
        raise SystemExit(f"{folder}: 拡張子を除いた名前が重なっている")
    return clips


def check_independent(clips: list[Clip], label: str) -> None:
    """同じ集合に同じ波形が 2 つ以上あれば止める（同じ入力の繰り返しを別の標本として数えない）"""
    seen: dict[str, str] = {}
    for c in clips:
        h = hashlib.sha256(np.ascontiguousarray(c.x).tobytes()).hexdigest()
        if h in seen:
            raise SystemExit(f"{label}: {seen[h]} と {c.name} の波形が同じ。1 つの入力からは 1 つの音だけを入れる")
        seen[h] = c.name


def align_lengths(sets: dict[str, list[Clip]], mode: str, common_sr: int, crop_sec: float | None) -> dict[str, int]:
    """長さをそろえる。名前ごとに使った長さ（common_sr のサンプル数）を返す"""
    if mode == "none":
        return {}
    if mode == "fixed":
        if crop_sec is None:
            raise SystemExit("--align fixed には --crop-sec が必要")
        n = int(round(crop_sec * common_sr))
        for label, clips in sets.items():
            for c in clips:
                if c.x.shape[0] < n:
                    raise SystemExit(f"{label}/{c.info['file']}: {c.x.shape[0] / common_sr:.3f} 秒しかなく、"
                                     f"--crop-sec {crop_sec} に足りない")
                c.x = c.x[:n]
        return {c.name: n for clips in sets.values() for c in clips}
    # shortest-by-name
    name_sets = {label: {c.name for c in clips} for label, clips in sets.items()}
    names = next(iter(name_sets.values()))
    for label, ns in name_sets.items():
        if ns != names:
            raise SystemExit(f"--align shortest-by-name: 名前の組が集合ごとに違う（{label}: "
                             f"足りない {sorted(names - ns)}、余分 {sorted(ns - names)}）")
    lengths = {n: min(c.x.shape[0] for clips in sets.values() for c in clips if c.name == n) for n in names}
    for clips in sets.values():
        for c in clips:
            c.x = c.x[: lengths[c.name]]
    return lengths


# MARK: - 埋め込み

def load_model(device: str):
    from kadtk.model_loader import PANNsModel
    ml = PANNsModel(MODEL_VARIANT)
    ml.device = torch.device(device)
    # 重みは CUDA の上で保存されているが、kadtk は map_location を付けずに torch.load するので、
    # 読み込みの間だけ CPU に置くよう差し替える（重みの値は変わらない）
    original_load = torch.load
    torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": "cpu"})
    try:
        ml.load_model()
    finally:
        torch.load = original_load
    return ml


def embed(ml, clip: Clip, common_sr: int) -> np.ndarray:
    """クリップ 1 つの埋め込み (2048,)。kadtk の get_embedding は float16 に丸めるので、その前の値を使う"""
    x = librosa.resample(clip.x, orig_sr=common_sr, target_sr=PANNS_SR, res_type="soxr_hq").astype(np.float32)
    clip.info.update({"panns_samples": x.shape[0], "duration_sec": x.shape[0] / PANNS_SR,
                      "peak": float(np.max(np.abs(x))), "rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))})
    with torch.no_grad():
        e = ml._get_embedding(x)
    e = e.detach().cpu().numpy().astype(np.float32)
    if e.shape != (1, 2048):
        raise RuntimeError(f"埋め込みの形状が想定と違う: {e.shape}")
    return e[0]


# MARK: - FAD

def frechet_distance(x: np.ndarray, y: np.ndarray) -> dict:
    """x (n_x × d), y (n_y × d) の埋め込みの FAD。float64 で計算し、内訳も返す

    元の式  FAD = ||μx − μy||² + Tr(Σx) + Tr(Σy) − 2 Tr((Σx Σy)^{1/2})
    Σx = Xcᵀ Xc / (n_x−1)、Σy = Ycᵀ Yc / (n_y−1)（Xc, Yc は中心化した埋め込み）とおく。
    Σx Σy は 2 つの半正定値行列の積なので固有値は実数で 0 以上であり、(Σx Σy)^{1/2} はその平方根を固有値に持つ
    行列（scipy の sqrtm が求める主平方根）である。よって Tr((Σx Σy)^{1/2}) = Σ_i sqrt(λ_i(Σx Σy))。
    さらに Σx Σy = Xcᵀ (Xc Ycᵀ Yc) / c（c = (n_x−1)(n_y−1)）で、AB と BA の 0 でない固有値は等しいので、
    λ(Σx Σy) の 0 でないものは λ(Xc Ycᵀ Yc Xcᵀ) / c = σ(M)² / c（M = Xc Ycᵀ、n_x × n_y）。
    したがって Tr((Σx Σy)^{1/2}) = Σ_i σ_i(M) / sqrt(c)（M の特異値の和）。
    Tr(Σx) = ||Xc||_F² / (n_x−1)。
    音の数が次元 2048 より少なく共分散が正則でないとき、sqrtm は丸め誤差で虚部や負の固有値の扱いの誤差を出すが、
    この方法は n_x × n_y の行列の特異値分解だけで済み、誤差が出にくい。照合用に sqrtm による値も返す。
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    nx, ny = x.shape[0], y.shape[0]
    if nx < 2 or ny < 2:
        raise SystemExit("集合には 2 つ以上の音が必要")
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mu_x, y - mu_y
    tr_cov_x = float(np.sum(xc ** 2) / (nx - 1))
    tr_cov_y = float(np.sum(yc ** 2) / (ny - 1))
    tr_covmean = float(np.linalg.svd(xc @ yc.T, compute_uv=False).sum() / np.sqrt((nx - 1) * (ny - 1)))
    mean_term = float(np.sum((mu_x - mu_y) ** 2))
    trace_term = tr_cov_x + tr_cov_y - 2 * tr_covmean

    # 照合用: 従来の sqrtm による値（float64）
    cov_x, cov_y = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    covmean, _ = scipy.linalg.sqrtm(cov_x @ cov_y, disp=False)
    imag_max = float(np.max(np.abs(covmean.imag))) if np.iscomplexobj(covmean) else 0.0
    fad_sqrtm = mean_term + float(np.trace(cov_x) + np.trace(cov_y) - 2 * np.trace(covmean.real))
    return {"fad": mean_term + trace_term, "mean_term": mean_term, "trace_term": trace_term,
            "fad_sqrtm": fad_sqrtm, "sqrtm_imag_max": imag_max}


def kadtk_frechet_distance(x: np.ndarray, y: np.ndarray, precision=torch.float32) -> float:
    """照合用: kadtk の実装での値。既定の float32 では sqrtm を float32 で行うため、音が少ないとずれる"""
    from kadtk.fad import calc_frechet_distance
    return float(calc_frechet_distance(x, y, (None, None), "cpu", precision=precision))


def max_cosine_between_files(emb: np.ndarray) -> float:
    if emb.shape[0] < 2:
        return float("nan")
    u = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    s = u @ u.T
    np.fill_diagonal(s, -1)
    return float(s.max())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# MARK: - main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=Path, required=True, help="参照音のフォルダ")
    parser.add_argument("--eval", action="append", required=True, metavar="NAME=DIR",
                        help="評価する合成音のフォルダ（複数可）")
    parser.add_argument("--common-sr", type=int, default=22050, help="PANNs の前にそろえる標本化周波数")
    parser.add_argument("--align", choices=["shortest-by-name", "fixed", "none"], default="shortest-by-name")
    parser.add_argument("--crop-sec", type=float, help="--align fixed で切る長さ（秒）")
    parser.add_argument("--keep-ref-length", action="store_true",
                        help="参照音は切らず、合成音の集合どうしだけ長さをそろえる（感度の確認用）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-embeddings", type=Path, help="埋め込みを npz で保存するフォルダ")
    parser.add_argument("--out", type=Path, help="結果の JSON")
    args = parser.parse_args()

    evals: list[tuple[str, Path]] = []
    for spec in args.eval:
        name, _, folder = spec.partition("=")
        if not folder or name == "ref":
            raise SystemExit(f"--eval は NAME=DIR の形で指定する（NAME は ref 以外）: {spec}")
        evals.append((name, Path(folder)))

    sets: dict[str, list[Clip]] = {"ref": load_set(args.ref, args.common_sr)}
    for name, folder in evals:
        sets[name] = load_set(folder, args.common_sr)
    for label, clips in sets.items():
        check_independent(clips, label)
    if args.keep_ref_length:
        # 参照音は切らず、合成音どうしだけ長さをそろえる（参照音が合成音と時間で対応しない場合の感度の確認用）
        lengths = align_lengths({k: v for k, v in sets.items() if k != "ref"}, args.align, args.common_sr, args.crop_sec)
    else:
        lengths = align_lengths(sets, args.align, args.common_sr, args.crop_sec)

    torch.set_grad_enabled(False)
    ml = load_model(args.device)
    embs = {label: np.stack([embed(ml, c, args.common_sr) for c in clips]) for label, clips in sets.items()}

    ref_emb = embs["ref"]
    results = []
    for name, folder in evals:
        emb = embs[name]
        fd = frechet_distance(ref_emb, emb)
        fd["fad_kadtk_float32"] = kadtk_frechet_distance(ref_emb, emb)
        results.append({"name": name, "dir": str(folder), "n": emb.shape[0],
                        "max_cosine_between_files": max_cosine_between_files(emb),
                        **fd, "files": [c.info for c in sets[name]]})
        print(f"{name}\tn={emb.shape[0]}\tFAD={fd['fad']:.4f}\t(mean {fd['mean_term']:.4f} + trace {fd['trace_term']:.4f})"
              f"\tsqrtm64={fd['fad_sqrtm']:.4f}\tkadtk32={fd['fad_kadtk_float32']:.4f}")
    if args.save_embeddings:
        args.save_embeddings.mkdir(parents=True, exist_ok=True)
        for label, clips in sets.items():
            np.savez(args.save_embeddings / f"{label}.npz", emb=embs[label], names=[c.name for c in clips])

    n_min = min(e.shape[0] for e in embs.values())
    if n_min <= 2048:
        print(f"注意: 集合の大きさ（最小 {n_min}）が埋め込みの次元 2048 以下なので、共分散は正則でない。"
              "FAD は集合の大きさに依存するので、同じ参照音の集合と同じ数の合成音どうしでだけ比べる", file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "method": {
                "metric": "FAD (Fréchet Audio Distance)",
                "embedding": f"PANNs {MODEL_VARIANT} (Wavegram_Logmel_Cnn14), 2048-d, clip-level, float32",
                "checkpoint": str(ml.model_file.name),
                "checkpoint_sha256": sha256_file(ml.model_file),
                "panns_sr": PANNS_SR,
                "common_sr": args.common_sr,
                "align": args.align,
                "keep_ref_length": args.keep_ref_length,
                "crop_sec": args.crop_sec,
                "aligned_samples_at_common_sr": lengths,
                "resampler": "librosa.resample(res_type='soxr_hq')",
                "amplitude_normalization": None,
                "covariance": "unbiased (n-1)",
                "stats_dtype": "float64",
                "trace_sqrt": "sum of singular values of Xc Yc^T / sqrt((nx-1)(ny-1)) (exact; no sqrtm)",
            },
            "env": {"python": platform.python_version(), "torch": torch.__version__, "librosa": librosa.__version__,
                    "kadtk": "1.1.0", "device": args.device, "platform": platform.platform()},
            "ref": {"dir": str(args.ref), "n": ref_emb.shape[0],
                    "max_cosine_between_files": max_cosine_between_files(ref_emb),
                    "files": [c.info for c in sets["ref"]]},
            "results": results,
        }
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
