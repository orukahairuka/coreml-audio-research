"""中間出力（ステージ tensor / 波形）の比較メトリクスと判定を共有するモジュール。

`compare_stages.py` と `compare_hifigan_isolation.py` から import して使う。
壊れている既存スクリプトには手を入れず、新規の調査スクリプト間でだけ共有する。

判定基準（cosine と rms_ratio = cur_rms / ref_rms のヒューリスティック）。
ステージ種別 kind で重みを変える:

- kind="tensor"（mel / encoder / postnet）: フレーム整列が効くので cosine を主に使う。
  - 破綻     : NaN/Inf、cosine < 0.5、または rms_ratio が [0.5, 2.0] の外
  - 大きな差 : cosine < 0.90、または rms_ratio が [0.7, 1.5] の外
  - 軽微な差 : cosine < 0.99、または rms_ratio が [0.9, 1.1] の外
  - 正常     : 上記以外

- kind="waveform"（HiFi-GAN 出力 / 最終波形）: 生サンプルの cosine は精度・エンジン差で
  位相がずれると簡単に下がり、破綻していなくても低く出る（例: int8×cpuAndGPU は MCD 上は
  良セルなのに f16 基準の波形 cosine が 0.17）。そのため **振幅 rms_ratio を主指標**に置き、
  cosine は参考扱いにする。
  - 破綻     : NaN/Inf、または rms_ratio が [0.5, 2.0] の外（振幅膨張・崩壊）
  - 大きな差 : rms_ratio が [0.7, 1.5] の外
  - 軽微な差 : rms_ratio が [0.9, 1.1] の外、または cosine < 0.90（位相/精度差。破綻ではない）
  - 正常     : 上記以外

しきい値はヒューリスティック。最終判断は聴感（wav）と MCD（DTW 整列）と合わせて行う。
ANE 破綻の本質は HiFi-GAN 出力で rms_ratio が ~4 倍に跳ねる点に出る。
"""

from __future__ import annotations

import numpy as np


def _flatten_aligned(ref: np.ndarray, cur: np.ndarray):
    """2 つの配列を squeeze→flatten し、長さを min に揃えて返す。

    自己回帰パイプラインは決定的なので通常は同形状だが、念のため min 長で切る。
    """
    a = np.asarray(ref).squeeze().astype(np.float64).reshape(-1)
    b = np.asarray(cur).squeeze().astype(np.float64).reshape(-1)
    n = min(a.size, b.size)
    return a[:n], b[:n], (a.size == b.size)


def basic_stats(arr: np.ndarray) -> dict:
    """単体配列の統計（リファレンスが無い段でも使う）。"""
    a = np.asarray(arr)
    flat = a.astype(np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    rms = float(np.sqrt(np.mean(finite * finite))) if finite.size else float("nan")
    return {
        "shape": tuple(int(x) for x in a.shape),
        "dtype": str(a.dtype),
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
        "peak": float(np.max(np.abs(finite))) if finite.size else float("nan"),
        "rms": rms,
        "has_nan": bool(np.isnan(flat).any()),
        "has_inf": bool(np.isinf(flat).any()),
    }


def compare(ref: np.ndarray, cur: np.ndarray, kind: str = "tensor") -> dict:
    """リファレンス ref に対する cur の差分メトリクスと判定を返す。

    kind: "tensor"（mel/encoder/postnet）か "waveform"（HiFi-GAN 出力/最終波形）。
    """
    a, b, same_shape = _flatten_aligned(ref, cur)
    cur_stats = basic_stats(cur)

    diff = b - a
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na > 0 and nb > 0:
        cosine = float(np.dot(a, b) / (na * nb))
    else:
        cosine = float("nan")

    ref_rms = float(np.sqrt(np.mean(a * a)))
    cur_rms = float(np.sqrt(np.mean(b * b)))
    rms_ratio = (cur_rms / ref_rms) if ref_rms > 0 else float("inf")

    verdict = judge(cosine, rms_ratio, cur_stats["has_nan"] or cur_stats["has_inf"], kind)

    out = {
        "same_shape": same_shape,
        "mae": mae,
        "rmse": rmse,
        "cosine": cosine,
        "ref_rms": ref_rms,
        "rms_ratio": rms_ratio,
        "verdict": verdict,
    }
    out.update(cur_stats)
    return out


def judge(cosine: float, rms_ratio: float, has_nan_inf: bool, kind: str = "tensor") -> str:
    """cosine / rms_ratio から 4 段階判定を返す。kind で cosine の重みを変える。"""
    if has_nan_inf:
        return "破綻"
    # 振幅崩壊・膨張は両 kind 共通で破綻
    if not (0.5 <= rms_ratio <= 2.0):
        return "破綻"

    if kind == "waveform":
        # 波形は振幅主、cosine は参考（位相/精度差で簡単に下がるため破綻には使わない）
        if not (0.7 <= rms_ratio <= 1.5):
            return "大きな差"
        cos_low = (not np.isnan(cosine)) and cosine < 0.90
        if cos_low or not (0.9 <= rms_ratio <= 1.1):
            return "軽微な差"
        return "正常"

    # tensor: cosine 主
    cos_bad_break = (not np.isnan(cosine)) and cosine < 0.5
    if cos_bad_break:
        return "破綻"
    cos_bad_large = (not np.isnan(cosine)) and cosine < 0.90
    if cos_bad_large or not (0.7 <= rms_ratio <= 1.5):
        return "大きな差"
    cos_bad_minor = (not np.isnan(cosine)) and cosine < 0.99
    if cos_bad_minor or not (0.9 <= rms_ratio <= 1.1):
        return "軽微な差"
    return "正常"


VERDICT_EMOJI = {
    "正常": "🟢 正常",
    "軽微な差": "🟡 軽微な差",
    "大きな差": "🟠 大きな差",
    "破綻": "🔴 破綻",
}


def md_table_header() -> list[str]:
    return [
        "| stage | condition | MAE | RMSE | cosine | min | max | mean | std | peak | rms | NaN/Inf | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]


def md_row(stage: str, condition: str, m: dict) -> str:
    nan_inf = "あり" if (m["has_nan"] or m["has_inf"]) else "なし"
    return (
        f"| {stage} | {condition} | {m['mae']:.4e} | {m['rmse']:.4e} | "
        f"{m['cosine']:.4f} | {m['min']:.4g} | {m['max']:.4g} | {m['mean']:.4g} | "
        f"{m['std']:.4g} | {m['peak']:.4g} | {m['rms']:.4g} | {nan_inf} | "
        f"{VERDICT_EMOJI.get(m['verdict'], m['verdict'])} |"
    )
