#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "matplotlib", "scipy"]
# ///
"""正式 RTF 測定（run_rtf_official.py）の記録を集計する（読むだけ。端末の記録は変更しない）

RTF は 1 回ごとに  rtf = rtf_wall_ms / 1000 / (T × 256 / 22050)  を計算し、それを平均する
（平均推論時間 ÷ 平均音声長 は使わない）。各入力の反復回数は同じなので、全反復の平均と入力ごとの平均の平均は一致する。

標準偏差は 3 種類を分けて出す:
  - sd_between_inputs: 入力ごとの平均の、入力間の標準偏差（入力長などによるばらつき。H200 の報告値と比べる対象）
  - sd_within_input_pooled: 同じ入力の反復間の標準偏差を、入力をまたいでプールしたもの sqrt(mean(var_i))
  - sd_all_iterations: 全反復をまとめた標準偏差（参考）

使い方
  uv run scripts/rtf_official/aggregate_rtf.py metrics/rtf_official/rtf_official_20261123
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SR, HOP = 22050, 256
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "eval/fad/data/formal_inputs/manifest.csv"
# H200 の sd は np.std（ddof=0）。iOS 側の比較欄も ddof=0 の入力間 sd を使う
H200 = {"time_mean_s": 1.06, "time_sd_s": 0.54, "rtf_mean": 0.317, "rtf_sd": 0.033}
LABEL_NAME = {"A": "A: FP32 (Enc GPU / Dec GPU padT / HiFi CPU)", "B": "B: FP16 (Enc GPU / Dec CPU / HiFi GPU)",
              "C": "C: INT8 W8A16 (all CPU)"}


def load(study: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    plan = json.loads((study / "plan.json").read_text())
    frames = [pd.read_csv(f, dtype={"error": str}) for f in sorted((study / "device").glob("iter_*.csv"))]
    it = pd.concat(frames, ignore_index=True)
    launches = []
    for f in sorted((study / "device").glob("launch_*.json")):
        info = json.loads(f.read_text())
        done = study / "device" / f.name.replace("launch_", "done_")
        info["completed"] = done.exists()
        if done.exists():
            info.update({f"end_{k}": v for k, v in json.loads(done.read_text()).items()})
        launches.append(info)
    la = pd.DataFrame(launches)
    # 計画のブロック・ブロック内の順番を付ける
    step_of = {(s["label"], s["block"]): s for s in plan["steps"]}
    la["block"] = la["launch"].str.extract(r"_b(\d+)_")[0].astype(int)
    la["slot_in_block"] = [step_of[(l, b)]["slot_in_block"] for l, b in zip(la["label"], la["block"])]
    la["step"] = [step_of[(l, b)]["step"] for l, b in zip(la["label"], la["block"])]
    it = it.merge(la[["launch", "block", "slot_in_block", "step"]], on="launch", how="left")
    return plan, it, la


def select_used(it: pd.DataFrame, repeats: int) -> pd.DataFrame:
    """(構成, 入力) ごとに、本測定が repeats 回そろった最後の起動の行だけを使う"""
    timed = it[(it["phase"] == "timed") & (it["status"] == "ok")]
    counts = timed.groupby(["label", "input", "launch"]).size().reset_index(name="n")
    full = counts[counts["n"] >= repeats].sort_values("launch").groupby(["label", "input"]).tail(1)
    key = set(zip(full["label"], full["input"], full["launch"]))
    it = it.copy()
    it["used"] = [(l, i, la) in key for l, i, la in zip(it["label"], it["input"], it["launch"])]
    return it


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", type=Path)
    ap.add_argument("--out", help="出力先のフォルダ名（study の中。既定 summary。途中経過は summary_interim_* などに分ける）")
    args = ap.parse_args()
    study = args.study
    out = study / (args.out or "summary")
    out.mkdir(exist_ok=True)
    plan, it, la = load(study)
    repeats = plan["repeats_per_input"]
    it = select_used(it, repeats)
    it["time_s"] = it["rtf_wall_ms"] / 1000
    it["dec_step_ms"] = it["decoder_wall_ms"] / it["T"]
    it.to_csv(out / "iterations_all.csv", index=False)
    checks = {}

    # 確認: T が台帳と同じ、出力の長さ、NaN、反復間で出力が同じか
    exp_T = {k: v["T"] for k, v in plan["inputs"].items()}
    ok_rows = it[it["status"] == "ok"]
    checks["T_mismatch_rows"] = int((ok_rows["T"] != ok_rows["input"].map(exp_T)).sum())
    checks["length_mismatch_rows"] = int((ok_rows["output_samples"] != ok_rows["T"] * HOP).sum())
    checks["nonfinite_rows"] = int((ok_rows["wave_nonfinite"] > 0).sum())
    checks["error_rows"] = int((it["status"] != "ok").sum())
    rms_var = ok_rows.groupby(["label", "input", "launch"])["wave_rms"].nunique()
    checks["inputs_with_rms_varying_across_reps"] = int((rms_var > 1).sum())
    checks["audio_sec_formula"] = "T*256/22050"

    used = it[it["used"] & (it["phase"] == "timed")].copy()
    warm = it[(it["phase"] == "warmup") & (it["status"] == "ok")
              & it.set_index(["label", "input", "launch"]).index.isin(
                  used.set_index(["label", "input", "launch"]).index)].copy()
    warm["time_s"] = warm["rtf_wall_ms"] / 1000

    # 入力ごと
    comp = ["encoder_wall_ms", "decoder_wall_ms", "hifigan_wall_ms", "deemph_ms",
            "encoder_predict_ms", "decoder_predict_ms", "hifigan_predict_ms"]
    g = used.groupby(["label", "input"])
    per_input = g.agg(T=("T", "first"), audio_sec=("audio_sec", "first"), n=("rtf", "size"),
                      block=("block", "first"), slot_in_block=("slot_in_block", "first"),
                      time_mean_s=("time_s", "mean"), time_sd_s=("time_s", "std"), time_median_s=("time_s", "median"),
                      time_min_s=("time_s", "min"), time_max_s=("time_s", "max"),
                      rtf_mean=("rtf", "mean"), rtf_sd=("rtf", "std"), rtf_median=("rtf", "median"),
                      footprint_peak_mb=("footprint_peak_mb", "max"),
                      **{f"{c}_mean": (c, "mean") for c in comp}).reset_index()
    per_input["time_cv"] = per_input["time_sd_s"] / per_input["time_mean_s"]
    wm = warm.groupby(["label", "input"])["time_s"].mean().rename("warmup_time_s")
    per_input = per_input.merge(wm, on=["label", "input"], how="left")
    per_input["warmup_over_timed"] = per_input["warmup_time_s"] / per_input["time_mean_s"]
    per_input.to_csv(out / "per_input.csv", index=False)

    # 構成ごと
    rows = []
    for label, d in used.groupby("label"):
        p = per_input[per_input["label"] == label]
        r = {"label": label, "config": LABEL_NAME.get(label, label), "inputs": len(p), "iterations": len(d),
             "repeats_per_input_min": int(p["n"].min()), "repeats_per_input_max": int(p["n"].max())}
        for name, col, pcol, psd in [("rtf", "rtf", "rtf_mean", "rtf_sd"), ("time_s", "time_s", "time_mean_s", "time_sd_s")]:
            r[f"{name}_mean"] = d[col].mean()
            r[f"{name}_median_iterations"] = d[col].median()
            r[f"{name}_median_of_input_means"] = p[pcol].median()
            r[f"{name}_sd_between_inputs"] = p[pcol].std(ddof=1)
            r[f"{name}_sd_within_input_pooled"] = float(np.sqrt((p[psd] ** 2).mean()))
            r[f"{name}_sd_all_iterations"] = d[col].std(ddof=1)
            # 開発者の synthesis_rtf.py は np.std（母標準偏差、ddof=0）で、入力ごとに 1 回の値の入力間のばらつき
            r[f"{name}_sd_between_inputs_ddof0"] = p[pcol].std(ddof=0)
            r[f"{name}_min"] = d[col].min()
            r[f"{name}_max"] = d[col].max()
        r["time_cv_within_input_median"] = p["time_cv"].median()
        r["time_cv_within_input_max"] = p["time_cv"].max()
        for c in comp:
            r[f"{c}_mean"] = d[c].mean()
        total = d["rtf_wall_ms"].mean()
        for c in ["encoder_wall_ms", "decoder_wall_ms", "hifigan_wall_ms", "deemph_ms"]:
            r[f"share_{c.replace('_ms', '')}"] = d[c].mean() / total
        r["other_ms_mean"] = (d["rtf_wall_ms"] - d[["encoder_wall_ms", "decoder_wall_ms", "hifigan_wall_ms", "deemph_ms"]]
                              .sum(axis=1)).mean()
        r["dec_step_ms_median"] = d["dec_step_ms"].median()
        r["mean_time_over_mean_audio_NOT_USED"] = d["time_s"].mean() / d["audio_sec"].mean()
        w = warm[warm["label"] == label]
        r["warmup_rtf_mean"] = w["rtf"].mean()
        r["warmup_over_timed_median"] = p["warmup_over_timed"].median()
        # 参考: 入力ごとの初回呼び出し（ウォームアップ行）を 1 入力 1 回の値として、開発者と同じ形（平均・ddof=0）で集計
        r["firstcall_rtf_mean"] = w["rtf"].mean()
        r["firstcall_rtf_sd_ddof0"] = w["rtf"].std(ddof=0)
        r["firstcall_time_s_mean"] = w["time_s"].mean()
        r["firstcall_time_s_sd_ddof0"] = w["time_s"].std(ddof=0)
        r["footprint_peak_mb_max"] = d["footprint_peak_mb"].max()
        r["footprint_mb_max"] = d["footprint_mb"].max()
        r["footprint_mb_median"] = d["footprint_mb"].median()
        for s in ["nominal", "fair", "serious", "critical"]:
            r[f"thermal_{s}_iterations"] = int((d["thermal_state"] == s).sum())
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "config_summary.csv", index=False)

    # T との関係（time = a + b T、rtf = a + b T）
    trows = []
    for label, p in per_input.groupby("label"):
        for y in ["time_mean_s", "rtf_mean", "decoder_wall_ms_mean"]:
            b, a = np.polyfit(p["T"], p[y], 1)
            pred = a + b * p["T"]
            r2 = 1 - ((p[y] - pred) ** 2).sum() / ((p[y] - p[y].mean()) ** 2).sum()
            rho = p[["T", y]].corr(method="spearman").iloc[0, 1]
            trows.append({"label": label, "y": y, "intercept": a, "slope_per_frame": b, "r2": r2, "spearman": rho})
        # 2 次まで（Decoder は 1 ステップの入力が伸びるので T^2 の項がありうる）
        c2, c1, c0 = np.polyfit(p["T"], p["time_mean_s"], 2)
        pred = c0 + c1 * p["T"] + c2 * p["T"] ** 2
        r2 = 1 - ((p["time_mean_s"] - pred) ** 2).sum() / ((p["time_mean_s"] - p["time_mean_s"].mean()) ** 2).sum()
        trows.append({"label": label, "y": "time_mean_s (quadratic)", "intercept": c0, "slope_per_frame": c1,
                      "quad_per_frame2": c2, "r2": r2})
        for lo, hi in [(0, 150), (150, 300), (300, 450), (450, 1000)]:
            q = p[(p["T"] >= lo) & (p["T"] < hi)]
            trows.append({"label": label, "y": f"T_bin [{lo},{hi})", "n_inputs": len(q),
                          "rtf_mean_bin": q["rtf_mean"].mean(), "time_mean_s_bin": q["time_mean_s"].mean()})
    pd.DataFrame(trows).to_csv(out / "t_relation.csv", index=False)

    # 実行順序の影響: 同じ入力の平均で割った時間（norm = 1 が平均）
    used["norm_time"] = used["time_s"] / used.groupby(["label", "input"])["time_s"].transform("mean")
    orows = []
    for label, d in used.groupby("label"):
        for factor in ["slot_in_block", "block", "rep", "position_in_launch", "thermal_state"]:
            for k, q in d.groupby(factor):
                orows.append({"label": label, "factor": factor, "level": k, "iterations": len(q),
                              "norm_time_mean": q["norm_time"].mean(), "rtf_mean": q["rtf"].mean()})
        # 起動の中での経過（起動内の通し番号）に対する傾き
        b, _ = np.polyfit(d["global_iter"], d["norm_time"], 1)
        orows.append({"label": label, "factor": "slope_norm_time_per_iteration_within_launch", "level": "",
                      "iterations": len(d), "norm_time_mean": b})
        # ブロック内の入力ごとに見た、反復の最初と最後の比
        first = d[d["rep"] == 0].set_index("input")["time_s"]
        last = d[d["rep"] == repeats - 1].set_index("input")["time_s"]
        orows.append({"label": label, "factor": f"rep{repeats - 1}_over_rep0_median", "level": "",
                      "iterations": len(first), "norm_time_mean": (last / first).median()})
    # ブロック・ブロック内の順番は入力ごとに 1 つしかないので、入力ごとの正規化では見えない。
    # 入力長の影響を T の 2 次式で除いた比（実測 / 予測）を、順番・ブロックごとに平均する（入力の違いと分けきれない点に注意）
    for label, p in per_input.groupby("label"):
        if len(p) < 4:
            continue
        coef = np.polyfit(p["T"], p["time_mean_s"], 2)
        resid = p["time_mean_s"] / np.polyval(coef, p["T"])
        for factor in ["slot_in_block", "block"]:
            for k, q in resid.groupby(p[factor]):
                orows.append({"label": label, "factor": f"{factor} (time / quadratic-T fit)", "level": k,
                              "iterations": int(p.loc[q.index, "n"].sum()), "norm_time_mean": q.mean()})
    pd.DataFrame(orows).to_csv(out / "order_effects.csv", index=False)

    # 3 構成とも測れた入力だけでそろえた比較（途中経過で入力の集合が違うときのため。全件そろえば全体と同じ）
    common = set.intersection(*[set(per_input.loc[per_input["label"] == l, "input"]) for l in per_input["label"].unique()])
    crow = []
    for label, p in per_input[per_input["input"].isin(common)].groupby("label"):
        crow.append({"label": label, "inputs": len(p), "time_mean_s": p["time_mean_s"].mean(),
                     "time_sd_between_inputs_ddof0": p["time_mean_s"].std(ddof=0),
                     "time_sd_within_input_pooled": float(np.sqrt((p["time_sd_s"] ** 2).mean())),
                     "rtf_mean": p["rtf_mean"].mean(), "rtf_sd_between_inputs_ddof0": p["rtf_mean"].std(ddof=0),
                     "rtf_sd_within_input_pooled": float(np.sqrt((p["rtf_sd"] ** 2).mean())),
                     "rtf_ratio_to_h200": p["rtf_mean"].mean() / H200["rtf_mean"]})
    pd.DataFrame(crow).to_csv(out / "common_inputs_summary.csv", index=False)

    # 起動ごと（開始時の状態・メモリ）
    lcols = ["step", "block", "slot_in_block", "label", "launch", "started_at", "thermal_state_at_launch",
             "thermal_wait_s", "thermal_state_after_wait", "battery_level", "battery_state", "low_power_mode",
             "footprint_after_load_mb", "completed", "end_thermal_state_at_end", "end_footprint_peak_mb", "end_timed_ok",
             "end_errors"]
    la["load_ms_total"] = la["load_ms"].apply(lambda m: sum(m.values()))
    la[[c for c in lcols if c in la.columns] + ["load_ms_total"]].sort_values("started_at").to_csv(out / "launches.csv", index=False)
    mem = used.groupby(["label", "launch"]).agg(footprint_mb_max=("footprint_mb", "max"),
                                                footprint_peak_mb=("footprint_peak_mb", "max"),
                                                footprint_mb_first=("footprint_mb", "first"),
                                                footprint_mb_last=("footprint_mb", "last")).reset_index()
    mem.to_csv(out / "memory_by_launch.csv", index=False)

    # H200 との比較
    hrows = [{"system": "NVIDIA H200 (developer, original PronounSE, PyTorch)", "inputs": 100,
              "time_mean_s": H200["time_mean_s"], "time_sd_s": H200["time_sd_s"], "rtf_mean": H200["rtf_mean"],
              "rtf_sd": H200["rtf_sd"], "rtf_ratio_to_h200": 1.0,
              "note": "1 pass per input after 1 global warm-up, np.std (ddof=0), chkpt 6000, CUDA sync before/after"}]
    for _, r in summary.iterrows():
        hrows.append({"system": f"iPhone 17 Pro {r['config']}", "inputs": r["inputs"],
                      "time_mean_s": r["time_s_mean"], "time_sd_s": r["time_s_sd_between_inputs_ddof0"],
                      "rtf_mean": r["rtf_mean"], "rtf_sd": r["rtf_sd_between_inputs_ddof0"],
                      "time_sd_within_input_pooled": r["time_s_sd_within_input_pooled"],
                      "firstcall_rtf_mean": r["firstcall_rtf_mean"],
                      "rtf_ratio_to_h200": r["rtf_mean"] / H200["rtf_mean"]})
    pd.DataFrame(hrows).to_csv(out / "h200_comparison.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))

    # 図
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for label, p in per_input.groupby("label"):
            ax[0].errorbar(p["T"], p["rtf_mean"], yerr=p["rtf_sd"], fmt="o", ms=3, color=colors.get(label), label=label)
            ax[1].errorbar(p["T"], p["time_mean_s"], yerr=p["time_sd_s"], fmt="o", ms=3, color=colors.get(label), label=label)
        ax[0].axhline(H200["rtf_mean"], color="gray", ls="--", label="H200 mean")
        ax[0].axhline(1.0, color="k", lw=0.5)
        ax[0].set(xlabel="T (frames)", ylabel="RTF", yscale="log", title="RTF vs T (mean ± SD over reps)")
        ax[1].set(xlabel="T (frames)", ylabel="inference time (s)", title="time vs T")
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(out / "fig_rtf_vs_T.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, d in used.groupby("label"):
            m = d.groupby("position_in_launch")["norm_time"].mean()
            ax.plot(m.index, m.values, "o-", color=colors.get(label), label=label)
        ax.axhline(1, color="k", lw=0.5)
        ax.set(xlabel="input position within launch", ylabel="time / input mean", title="drift within launch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_order_drift.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # 図が作れなくても集計は残す
        print(f"図を作れなかった: {e}")

    write_time_tables(out, per_input, used, summary, repeats)

    pd.set_option("display.width", 200)
    print(json.dumps(checks, ensure_ascii=False))
    print(summary[["label", "inputs", "iterations", "rtf_mean", "rtf_sd_between_inputs", "rtf_sd_within_input_pooled",
                   "time_s_mean", "time_s_sd_between_inputs", "time_s_sd_within_input_pooled",
                   "footprint_peak_mb_max"]].to_string(index=False))
    return 0


def write_time_tables(out: Path, per_input: pd.DataFrame, used: pd.DataFrame, summary: pd.DataFrame, repeats: int) -> None:
    """指導教員の要望（2026-11-23）: 入力ごとの素の推論時間（秒）の平均・標準偏差・回数と、入力音声長との関係

    RTF とは別のファイルに分ける。標準偏差は、入力ごとの表では同じ入力の反復間（ddof=1）、
    全体の表では入力間（入力ごとの平均のばらつき）と反復間（プール）を分けて出す。
    """
    man = pd.read_csv(MANIFEST)
    man = man[man["speaker"] == "m-00"].copy()
    man["input"] = man["output"].map(lambda x: Path(x).stem)
    man["input_duration_s"] = man["source_samples"] / man["source_sr"]   # 元の口真似音（48 kHz）の長さ
    man["input_duration_22050_s"] = man["samples_22050"] / SR             # 22050 Hz に変換した入力の長さ
    base = man[["input", "reference", "input_duration_s", "input_duration_22050_s", "expected_T"]].rename(
        columns={"expected_T": "T"})
    base["output_audio_s"] = base["T"] * HOP / SR

    def wide(cols: dict[str, str]) -> pd.DataFrame:
        table = base.copy()
        for label in ["A", "B", "C"]:
            p = per_input[per_input["label"] == label].set_index("input")
            for new, old in cols.items():
                table[f"{label}_{new}"] = table["input"].map(p[old]) if len(p) else np.nan
        return table.sort_values("input_duration_s").reset_index(drop=True)

    # 1・2・4: 入力 100 件ごとの推論時間（秒）
    time_wide = wide({"time_mean_s": "time_mean_s", "time_sd_s": "time_sd_s", "n": "n",
                      "time_median_s": "time_median_s", "time_min_s": "time_min_s", "time_max_s": "time_max_s",
                      "time_cv": "time_cv"})
    time_wide.to_csv(out / "per_input_time_s.csv", index=False)
    # 5: RTF は別のファイル
    rtf_wide = wide({"rtf_mean": "rtf_mean", "rtf_sd": "rtf_sd", "n": "n"})
    rtf_wide.to_csv(out / "per_input_rtf.csv", index=False)

    # 6: 全体の表（卒論本文用）。時間と RTF を分ける
    def overall(metric: str, unit_col: str, sd_col: str) -> pd.DataFrame:
        rows = []
        for _, r in summary.iterrows():
            p = per_input[per_input["label"] == r["label"]]
            rows.append({
                "label": r["label"], "config": r["config"], "inputs": len(p), "repeats_per_input": repeats,
                "iterations": int(p["n"].sum()),
                f"{metric}_mean": p[unit_col].mean(),
                f"{metric}_median_of_input_means": p[unit_col].median(),
                f"{metric}_sd_between_inputs_ddof1": p[unit_col].std(ddof=1),
                f"{metric}_sd_between_inputs_ddof0": p[unit_col].std(ddof=0),
                f"{metric}_sd_within_input_pooled": float(np.sqrt((p[sd_col] ** 2).mean())),
                f"{metric}_sd_within_input_median": p[sd_col].median(),
                f"{metric}_sd_within_input_max": p[sd_col].max(),
                f"{metric}_min_input_mean": p[unit_col].min(),
                f"{metric}_max_input_mean": p[unit_col].max(),
            })
        return pd.DataFrame(rows)
    overall("time_s", "time_mean_s", "time_sd_s").to_csv(out / "overall_time_s.csv", index=False)
    overall("rtf", "rtf_mean", "rtf_sd").to_csv(out / "overall_rtf.csv", index=False)

    # 入力音声長と推論時間の関係（time = a + b × 長さ）
    rows = []
    for label in ["A", "B", "C"]:
        q = time_wide[["input_duration_s", f"{label}_time_mean_s"]].dropna()
        if len(q) < 3:
            continue
        x, y = q["input_duration_s"], q[f"{label}_time_mean_s"]
        b, a = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"label": label, "inputs": len(q), "intercept_s": a, "slope_s_per_input_s": b, "r2": r2,
                     "pearson": x.corr(y), "spearman": x.corr(y, method="spearman")})
    pd.DataFrame(rows).to_csv(out / "time_vs_input_duration_fit.csv", index=False)

    # 3・4: 散布図（入力音声長 × 平均推論時間、誤差棒は同じ入力の 10 回の標準偏差）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
        names = {"A": "FP32", "B": "FP16", "C": "INT8"}
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, label in zip(axes, ["A", "B", "C"]):
            x, y, e = (time_wide["input_duration_s"], time_wide[f"{label}_time_mean_s"], time_wide[f"{label}_time_sd_s"])
            ax.errorbar(x, y, yerr=e, fmt="o", ms=3, capsize=2, color=colors[label], ecolor="gray", elinewidth=0.8)
            ax.set(xlabel="input duration (s)", ylabel="inference time (s)", title=f"{label}: {names[label]}")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
        fig.suptitle("Mean inference time per input (error bar: SD over repeats)")
        fig.tight_layout()
        fig.savefig(out / "fig_time_vs_input_duration_panels.png", dpi=200)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for label in ["A", "B", "C"]:
            ax.errorbar(time_wide["input_duration_s"], time_wide[f"{label}_time_mean_s"],
                        yerr=time_wide[f"{label}_time_sd_s"], fmt="o", ms=3, capsize=2, color=colors[label],
                        label=f"{label}: {names[label]}")
        ax.set(xlabel="input duration (s)", ylabel="inference time (s)", yscale="log",
               title="Mean inference time per input (log scale)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_time_vs_input_duration.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"図を作れなかった: {e}")


if __name__ == "__main__":
    sys.exit(main())
