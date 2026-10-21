#!/usr/bin/env python3
"""卒論第4章 4-2 節の表と図を作る。

run_thesis_bench.py の出力（<計測ディレクトリ>/raw/）と、thesis_ch4_metrics.py が作る
*_runs_metrics.csv を読み、<計測ディレクトリ>/tables/ と figures/ に書き出す。
数値は「中央値 [最小〜最大] (n)」で示す。失敗した実行は除かずに件数として表に出す。

  tables/4-2-1_load.{csv,md}            (1) モデルの読み込み時間
  tables/4-2-2_inference.{csv,md}       (2) 推論時間・RTF
  tables/4-2-2_inference_by_input.csv        入力ごとの内訳
  tables/4-2-3-4_waveform.{csv,md}      (3)(4) 12 条件の合成音の比較（基準 Float32 + cpuOnly）
  tables/4-2-3-4_vocoder.{csv,md}       (3)(4) HiFi-GAN だけの比較（基準 PyTorch）
  tables/4-2-5_e2e.{csv,md}             (5) End-to-End 時間
  tables/run_status.{csv,md}                 起動と実行の成否（全フェーズ）
  tables/compute_plan.{csv,md}               MLComputePlan の計画上の割り当て（参考）
  figures/*.png

実行例:
    PronounSE/venv/bin/python scripts/aggregate_thesis_ch4.py metrics/thesis_ch4_iPhone_3_<日時>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PRECISIONS = ["Float32", "Float16", "Int8"]
COMPUTE_UNITS = ["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"]
CONDITIONS = [(p, cu) for cu in COMPUTE_UNITS for p in PRECISIONS]
INPUTS = ["1", "6", "11", "2"]

# dataviz の参照パレット（ライト）。categorical は固定順で使う
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#9ec1ee", "#2a78d6", "#0d3a73"])

plt.rcParams.update({
    "font.family": "Hiragino Sans",
    "axes.unicode_minus": False,
    "savefig.dpi": 200,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2,
    "ytick.color": TEXT2,
    "axes.titlecolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------- 読み込み

def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def cond_label(p: str, cu: str) -> str:
    return f"{p} × {cu}"


# ---------------------------------------------------------------- 書式

def summarize(values) -> dict:
    a = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if a.size == 0:
        return {"median": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def fmt(x: float, digits: int = 3) -> str:
    """有効数字 digits 桁。1000 以上は整数で桁区切り"""
    if x is None or math.isnan(x):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}g}"


def fmt_s(s: dict, digits: int = 3, show_n: bool = True) -> str:
    if s["n"] == 0:
        return "—"
    if s["n"] == 1 or s["min"] == s["max"]:
        body = fmt(s["median"], digits)
    else:
        body = f"{fmt(s['median'], digits)} [{fmt(s['min'], digits)}〜{fmt(s['max'], digits)}]"
    return f"{body} (n={s['n']})" if show_n else body


def write_table(out: Path, name: str, header: list[str], rows: list[list], md_rows: list[list] | None = None,
                note: str = "") -> None:
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    if md_rows is not None:
        lines = ["| " + " | ".join(md_rows[0]) + " |", "|" + "---|" * len(md_rows[0])]
        lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in md_rows[1:]]
        if note:
            lines += ["", note]
        (out / f"{name}.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 表

def table_status(out: Path, host: list[dict], infer: list[dict], vocoder: list[dict], load: list[dict]) -> None:
    by = defaultdict(lambda: defaultdict(int))
    for r in host:
        if r["mode"] == "thermal":
            continue
        by[(r["mode"], r["precision"], r["compute_units"])][f"launch_{r['status']}"] += 1
    for mode, rows in (("load", load), ("infer", infer), ("vocoder", vocoder)):
        for r in rows:
            by[(mode, r["precision"], r["compute_units"])][f"run_{r['status']}"] += 1
    keys = sorted({k for d in by.values() for k in d})
    header = ["mode", "precision", "compute_units"] + keys
    rows = []
    for mode in ("load", "plan", "infer", "vocoder"):
        for p, cu in CONDITIONS:
            d = by.get((mode, p, cu))
            if d:
                rows.append([mode, p, cu] + [d.get(k, 0) for k in keys])
    md = [["モード", "条件"] + keys] + [[r[0], cond_label(r[1], r[2])] + r[3:] for r in rows]
    write_table(out, "run_status", header, rows, md,
                "launch_* は PC 側から見た起動の結果（ok / failed / crashed / timeout）、run_* はアプリが書いた実行ごとの行の status。")


def table_load(out: Path, load: list[dict], host: list[dict]) -> dict:
    kind_of = {r["launch_id"]: r.get("load_kind", "") for r in host}
    groups = defaultdict(lambda: defaultdict(list))
    for r in load:
        if r["status"] != "success":
            continue
        key = (r["precision"], r["compute_units"])
        if r["kind"] == "first":
            k = kind_of.get(r["launch_id"], "") or "first"
            groups[key][k].append(r)
        else:
            groups[key]["reload"].append(r)
    header = ["precision", "compute_units", "kind", "n", "total_ms_median", "total_ms_min", "total_ms_max",
              "encoder_ms_median", "decoder_ms_median", "hifigan_ms_median"]
    rows, md = [], [["条件", "インストール直後の初回", "起動し直しただけの初回", "再読み込み",
                     "インストール直後の内訳 Enc / Dec / HiFi-GAN（中央値）"]]
    summary = {}
    for p, cu in CONDITIONS:
        g = groups.get((p, cu), {})
        cells = []
        for kind in ("after_install", "restart", "reload"):
            rs = g.get(kind, [])
            s = summarize([fnum(r["total_ms"]) for r in rs])
            parts = {m: summarize([fnum(r[f"{m}_ms"]) for r in rs])["median"] for m in ("encoder", "decoder", "hifigan")}
            rows.append([p, cu, kind, s["n"], s["median"], s["min"], s["max"],
                         parts["encoder"], parts["decoder"], parts["hifigan"]])
            cells.append(fmt_s(s, 3))
            summary[(p, cu, kind)] = s
            if kind == "after_install":
                breakdown = " / ".join(fmt(parts[m], 3) for m in ("encoder", "decoder", "hifigan"))
        md.append([cond_label(p, cu)] + cells + [breakdown])
    write_table(out, "4-2-1_load", header, rows, md,
                "単位は ms。3 モデル（Encoder・Decoder・HiFi-GAN）の読み込みの合計。中央値 [最小〜最大] (n)。\n"
                "インストール直後: アプリを入れ直してから起動した直後の読み込み。起動し直しただけ: 入れ直さずに"
                "プロセスだけ起動し直した直後の読み込み（Core ML がディスクに残した準備結果を再利用する）。"
                "再読み込み: 同じプロセスでモデルを解放してから読み込み直したもの。")
    return summary


def table_inference(out: Path, infer: list[dict]) -> dict:
    ok = [r for r in infer if r["status"] == "success"]
    header = ["precision", "compute_units", "metric", "phase", "n", "median", "min", "max"]
    rows = []
    md = [["条件", "初回の推論 合成時間 [ms]（入力1）", "定常時 合成時間 [ms]（入力1, T′=262）",
           "定常時 RTF（4入力）", "Decoder 1ステップ [ms]（定常時の各実行の中央値）",
           "Encoder [ms]（定常時）", "HiFi-GAN [ms]（定常時）"]]
    summary = {}
    for p, cu in CONDITIONS:
        rs = [r for r in ok if r["precision"] == p and r["compute_units"] == cu]
        steady = [r for r in rs if r["phase"] == "steady"]
        initial = [r for r in rs if r["phase"] == "initial"]
        in1 = [r for r in steady if r["input_name"] == INPUTS[0]]
        metrics = {
            ("synth_ms", "initial"): summarize([fnum(r["synth_ms"]) for r in initial]),
            ("synth_ms", "steady_input1"): summarize([fnum(r["synth_ms"]) for r in in1]),
            ("rtf", "steady"): summarize([fnum(r["rtf"]) for r in steady]),
            ("rtf", "initial"): summarize([fnum(r["rtf"]) for r in initial]),
            ("decoder_step_ms_median", "steady"): summarize([fnum(r["decoder_step_ms_median"]) for r in steady]),
            ("encoder_ms", "steady"): summarize([fnum(r["encoder_ms"]) for r in steady]),
            ("hifigan_ms", "steady"): summarize([fnum(r["hifigan_ms"]) for r in steady]),
            ("decoder_total_ms", "steady_input1"): summarize([fnum(r["decoder_total_ms"]) for r in in1]),
        }
        for (m, ph), s in metrics.items():
            rows.append([p, cu, m, ph, s["n"], s["median"], s["min"], s["max"]])
        summary[(p, cu)] = metrics
        md.append([cond_label(p, cu),
                   fmt_s(metrics[("synth_ms", "initial")], 4, False),
                   fmt_s(metrics[("synth_ms", "steady_input1")], 4),
                   fmt_s(metrics[("rtf", "steady")], 3),
                   fmt_s(metrics[("decoder_step_ms_median", "steady")], 3),
                   fmt_s(metrics[("encoder_ms", "steady")], 3),
                   fmt_s(metrics[("hifigan_ms", "steady")], 3)])
    write_table(out, "4-2-2_inference", header, rows, md,
                "合成時間 = Encoder 1 回 + Decoder T′ 回 + HiFi-GAN 1 回の prediction(from:) の合計。"
                "RTF = 合成時間 ÷ (256T′ / 22050 秒)。初回の推論は読み込み直後の 1 回目（入力1）。"
                "定常時は入力ごとにウォームアップ 1 回の後の 5 回。")

    # 入力ごとの内訳
    header2 = ["precision", "compute_units", "input_name", "T", "T_prime", "phase", "n",
               "synth_ms_median", "synth_ms_min", "synth_ms_max", "rtf_median", "rtf_min", "rtf_max",
               "e2e_ms_median", "thermal_before_states"]
    rows2 = []
    for p, cu in CONDITIONS:
        for name in INPUTS:
            for ph in ("initial", "warmup", "steady"):
                rs = [r for r in ok if r["precision"] == p and r["compute_units"] == cu
                      and r["input_name"] == name and r["phase"] == ph]
                if not rs:
                    continue
                s = summarize([fnum(r["synth_ms"]) for r in rs])
                rt = summarize([fnum(r["rtf"]) for r in rs])
                e = summarize([fnum(r["e2e_ms"]) for r in rs])
                states = ",".join(sorted({r["thermal_before"] for r in rs}))
                rows2.append([p, cu, name, rs[0]["T"], rs[0]["T_prime"], ph, s["n"], s["median"], s["min"], s["max"],
                              rt["median"], rt["min"], rt["max"], e["median"], states])
    write_table(out, "4-2-2_inference_by_input", header2, rows2)
    return summary


WAVE_METRICS = [
    ("deemph_over1_ratio", "範囲の超過率（|x|>1, デエンファシス後）"),
    ("deemph_max", "最大値（デエンファシス後）"),
    ("deemph_min", "最小値（デエンファシス後）"),
    ("hifigan_rms_ratio", "RMS比（HiFi-GAN 出力）"),
    ("hifigan_peak_ratio", "Peak比（HiFi-GAN 出力）"),
    ("hifigan_cosine", "コサイン類似度（HiFi-GAN 出力）"),
    ("deemph_rms_ratio", "RMS比（デエンファシス後）"),
    ("deemph_peak_ratio", "Peak比（デエンファシス後）"),
    ("deemph_cosine", "コサイン類似度（デエンファシス後）"),
    ("wav_saturation_ratio", "WAV の飽和率（±32767/−32768）"),
    ("hifigan_nan", "NaN の数（HiFi-GAN 出力）"),
    ("hifigan_inf", "Inf の数（HiFi-GAN 出力）"),
    ("decoder_mel_cosine", "Decoder 出力メルのコサイン類似度"),
]


def table_waveform(out: Path, rows_in: list[dict], name: str, ref_note: str) -> dict:
    """条件 × 入力ごとに、定常時の実行の指標を集計する"""
    ok = [r for r in rows_in if r["status"] == "success" and r["phase"] == "steady"]
    metrics = [(m, label) for m, label in WAVE_METRICS if ok and m in ok[0]]
    header = ["precision", "compute_units", "input_name", "T", "T_prime", "n", "distinct_hifigan_sha256"]
    for m, _ in metrics:
        header += [f"{m}_median", f"{m}_min", f"{m}_max"]
    rows = []
    table: dict = {}
    for p, cu in CONDITIONS:
        for inp in INPUTS:
            rs = [r for r in ok if r["precision"] == p and r["compute_units"] == cu and r["input_name"] == inp]
            if not rs:
                continue
            row = [p, cu, inp, rs[0]["T"], rs[0]["T_prime"], len(rs), len({r["hifigan_wave_sha256"] for r in rs})]
            for m, _ in metrics:
                s = summarize([fnum(r.get(m)) for r in rs])
                table[(p, cu, inp, m)] = s
                row += [s["median"], s["min"], s["max"]]
            rows.append(row)
    write_table(out, name, header, rows)

    # md: 指標ごとに 条件 × 入力 の行列
    lines = [f"基準: {ref_note}。値は定常時 5 回の中央値（5 回で値が違う場合は [最小〜最大]）。", ""]
    for m, label in metrics:
        lines += [f"### {label}", "", "| 条件 | " + " | ".join(f"入力{i}" for i in INPUTS) + " |",
                  "|---|" + "---|" * len(INPUTS)]
        for p, cu in CONDITIONS:
            cells = [fmt_s(table.get((p, cu, i, m), summarize([])), 4, False) for i in INPUTS]
            lines.append(f"| {cond_label(p, cu)} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / f"{name}.md").write_text("\n".join(lines))
    return table


def table_e2e(out: Path, infer: list[dict]) -> dict:
    ok = [r for r in infer if r["status"] == "success" and r["phase"] == "steady"]
    header = ["precision", "compute_units", "input_name", "n", "e2e_ms_median", "e2e_ms_min", "e2e_ms_max",
              "preprocess_ms_median", "synth_ms_median", "deemph_ms_median", "other_ms_median"]
    rows = []
    md = [["条件", "E2E [ms]（入力1）", "前処理", "合成（モデル呼び出し）", "デエンファシス",
           "それ以外（入力配列の準備など）", "E2E ÷ 音声の長さ"]]
    summary = {}
    for p, cu in CONDITIONS:
        for inp in INPUTS:
            rs = [r for r in ok if r["precision"] == p and r["compute_units"] == cu and r["input_name"] == inp]
            if not rs:
                continue
            e = summarize([fnum(r["e2e_ms"]) for r in rs])
            parts = {k: summarize([fnum(r[k]) for r in rs])["median"]
                     for k in ("preprocess_ms", "synth_ms", "deemph_ms")}
            other = summarize([fnum(r["e2e_ms"]) - fnum(r["preprocess_ms"]) - fnum(r["synth_ms"]) - fnum(r["deemph_ms"])
                               for r in rs])["median"]
            rows.append([p, cu, inp, e["n"], e["median"], e["min"], e["max"],
                         parts["preprocess_ms"], parts["synth_ms"], parts["deemph_ms"], other])
            summary[(p, cu, inp)] = {"e2e": e, **parts, "other": other}
            if inp == INPUTS[0]:
                audio = summarize([fnum(r["audio_sec"]) for r in rs])["median"] * 1000
                md.append([cond_label(p, cu), fmt_s(e, 4), fmt(parts["preprocess_ms"], 3),
                           fmt(parts["synth_ms"], 4), fmt(parts["deemph_ms"], 2), fmt(other, 4),
                           fmt(e["median"] / audio, 3)])
    write_table(out, "4-2-5_e2e", header, rows, md,
                "E2E = 入力ファイルの読み込みと前処理の開始から、デエンファシスを適用した波形が得られるまで"
                "（npy / wav の保存は含まない）。定常時 5 回。それ以外 = E2E − 前処理 − 合成 − デエンファシス"
                "（各モデルへの入力配列の準備・出力の取り出しなど、モデルの呼び出し以外の時間）。")
    return summary


def table_compute_plan(out: Path, device_dir: Path) -> None:
    rows = []
    for p, cu in CONDITIONS:
        d = device_dir / "compute_plan" / f"{p}_{cu}"
        for model in ("Encoder", "Decoder", "HiFiGAN"):
            files = list(d.glob(f"*{model}*.json")) if d.is_dir() else []
            if not files:
                rows.append([p, cu, model, "", "", "", "", ""])
                continue
            j = json.loads(files[0].read_text())
            s = j["dispatchSummary"]
            rows.append([p, cu, model, j["operationCount"], s.get("cpu", 0), s.get("gpu", 0),
                         s.get("neuralEngine", 0), s.get("unknown", 0)])
    header = ["precision", "compute_units", "model", "ops", "cpu", "gpu", "neural_engine", "unknown"]
    md = [["条件", "モデル", "op 数", "CPU", "GPU", "Neural Engine", "割り当てなし（const 等）"]]
    md += [[cond_label(r[0], r[1]), r[2]] + r[3:] for r in rows]
    write_table(out, "compute_plan", header, rows, md,
                "MLComputePlan による実行前の計画上の割り当て（参考）。実際にどの計算ユニットで演算されたかではない。")


# ---------------------------------------------------------------- 図

def _cond_ticks(ax, horizontal: bool = True):
    labels = [f"{p}\n{cu}" for p, cu in CONDITIONS]
    if horizontal:
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(labels, fontsize=7)
    for i in (2.5, 5.5, 8.5):
        (ax.axvline if horizontal else ax.axhline)(i, color=GRID, lw=1, zorder=0)


def _range_marks(ax, xs, stats, color, label):
    med = [s["median"] for s in stats]
    lo = [s["median"] - s["min"] if s["n"] else 0 for s in stats]
    hi = [s["max"] - s["median"] if s["n"] else 0 for s in stats]
    ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", ms=5, color=color, ecolor=color, elinewidth=1.5,
                capsize=3, label=label, zorder=3)


def fig_load(out: Path, s: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    labels = {"after_install": "インストール直後の初回", "restart": "起動し直しただけの初回", "reload": "再読み込み"}
    for k, (kind, color) in enumerate(zip(labels, SERIES)):
        xs = np.arange(len(CONDITIONS)) + (k - 1) * 0.22
        _range_marks(ax, xs, [s.get((p, cu, kind), summarize([])) for p, cu in CONDITIONS], color, labels[kind])
    ax.set_yscale("log")
    ax.set_ylabel("3 モデルの読み込み時間 [ms]（対数）")
    ax.grid(axis="y", color=GRID, lw=0.8)
    _cond_ticks(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.set_title("モデルの読み込み時間（中央値と最小〜最大）", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "4-2-1_load.png")
    plt.close(fig)


def fig_rtf(out: Path, s: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(CONDITIONS))
    _range_marks(ax, xs - 0.12, [s[(p, cu)][("rtf", "steady")] for p, cu in CONDITIONS], SERIES[0], "定常時（4 入力 × 5 回）")
    _range_marks(ax, xs + 0.12, [s[(p, cu)][("rtf", "initial")] for p, cu in CONDITIONS], SERIES[1], "初回の推論（入力1）")
    ax.axhline(1.0, color=TEXT2, lw=1, ls="--", zorder=1)
    ax.text(len(CONDITIONS) - 0.5, 1.0, "RTF = 1", va="bottom", ha="right", fontsize=8, color=TEXT2)
    ax.set_ylabel("RTF（合成時間 ÷ 音声の長さ）")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    _cond_ticks(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("合成の RTF（中央値と最小〜最大）", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "4-2-2_rtf.png")
    plt.close(fig)


def fig_heatmap(out: Path, table: dict, metric: str, title: str, fname: str, fmt_digits: int = 3,
                percent: bool = False) -> None:
    data = np.array([[table.get((p, cu, i, metric), summarize([]))["median"] for i in INPUTS] for p, cu in CONDITIONS])
    if np.all(np.isnan(data)):
        return
    shown = data * 100 if percent else data
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    vmax = np.nanmax(shown) if np.nanmax(shown) > 0 else 1
    im = ax.imshow(shown, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
    for (y, x), v in np.ndenumerate(shown):
        if np.isnan(v):
            txt = "—"
        else:
            txt = f"{v:.{fmt_digits}g}" + ("%" if percent else "")
        dark = not np.isnan(v) and v > vmax * 0.55
        ax.text(x, y, txt, ha="center", va="center", fontsize=8, color="white" if dark else TEXT)
    ax.set_xticks(range(len(INPUTS)))
    ax.set_xticklabels([f"入力{i}" for i in INPUTS], fontsize=8)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([cond_label(p, cu) for p, cu in CONDITIONS], fontsize=8)
    for i in (2.5, 5.5, 8.5):
        ax.axhline(i, color="white", lw=2)
    ax.set_title(title, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(out / fname)
    plt.close(fig)


def fig_e2e(out: Path, s: dict) -> None:
    keys = [(p, cu) for p, cu in CONDITIONS if (p, cu, INPUTS[0]) in s]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(keys))
    synth = np.array([s[(p, cu, INPUTS[0])]["synth_ms"] for p, cu in keys]) / 1000
    other = np.array([s[(p, cu, INPUTS[0])]["other"] for p, cu in keys]) / 1000
    pre = np.array([s[(p, cu, INPUTS[0])]["preprocess_ms"] + s[(p, cu, INPUTS[0])]["deemph_ms"] for p, cu in keys]) / 1000
    w = 0.6
    ax.bar(xs, synth, w, color=SERIES[0], label="合成（モデルの呼び出し）", edgecolor="white", linewidth=1)
    ax.bar(xs, other, w, bottom=synth, color=SERIES[1], label="それ以外（入力配列の準備など）", edgecolor="white", linewidth=1)
    ax.bar(xs, pre, w, bottom=synth + other, color=SERIES[2], label="前処理とデエンファシス", edgecolor="white", linewidth=1)
    audio = 256 * 262 / 22050
    ax.axhline(audio, color=TEXT2, lw=1, ls="--")
    ax.text(len(keys) - 0.5, audio, f"音声の長さ {audio:.2f} 秒", va="bottom", ha="right", fontsize=8, color=TEXT2)
    ax.set_ylabel("時間 [秒]（入力1, 定常時の中央値）")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p}\n{cu}" for p, cu in keys], fontsize=7)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("End-to-End 時間の内訳", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "4-2-5_e2e.png")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench_dir", type=Path, help="metrics/thesis_ch4_<端末>_<日時>（raw/ を含む）")
    args = parser.parse_args()
    raw = args.bench_dir / "raw"
    device = raw / "device"
    tables = args.bench_dir / "tables"
    figures = args.bench_dir / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    host = read_csv(raw / "host_runs.csv")
    load = read_csv(device / "load_runs.csv")
    infer = read_csv(raw / "infer_runs_metrics.csv") or read_csv(device / "infer_runs.csv")
    vocoder = read_csv(raw / "vocoder_runs_metrics.csv") or read_csv(device / "vocoder_runs.csv")

    table_status(tables, host, infer, vocoder, load)
    if load:
        fig_load(figures, table_load(tables, load, host))
    if infer:
        fig_rtf(figures, table_inference(tables, infer))
        wave = table_waveform(tables, infer, "4-2-3-4_waveform", "同じ入力の Float32 + cpuOnly（定常時の 1 回目）")
        fig_heatmap(figures, wave, "deemph_over1_ratio", "範囲の超過率 |x|>1（デエンファシス後）",
                    "4-2-3-4_over1_ratio.png", 3, percent=True)
        fig_heatmap(figures, wave, "hifigan_rms_ratio", "RMS比（HiFi-GAN 出力, 基準 Float32 × cpuOnly）",
                    "4-2-3-4_hifigan_rms_ratio.png", 3)
        fig_heatmap(figures, wave, "decoder_mel_cosine", "Decoder 出力メルのコサイン類似度",
                    "4-2-3-4_decoder_mel_cosine.png", 4)
        fig_e2e(figures, table_e2e(tables, infer))
    if vocoder:
        voc = table_waveform(tables, vocoder, "4-2-3-4_vocoder", "同じメルを入れた PyTorch 版 HiFi-GAN の出力")
        fig_heatmap(figures, voc, "hifigan_rms_ratio", "HiFi-GAN だけの比較：RMS比（基準 PyTorch）",
                    "4-2-3-4_vocoder_rms_ratio.png", 3)
        fig_heatmap(figures, voc, "deemph_over1_ratio", "HiFi-GAN だけの比較：範囲の超過率 |x|>1",
                    "4-2-3-4_vocoder_over1_ratio.png", 3, percent=True)
    if device.is_dir():
        table_compute_plan(tables, device)
    print(f"# tables → {tables}")
    print(f"# figures → {figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
