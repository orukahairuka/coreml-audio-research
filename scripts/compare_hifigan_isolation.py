#!/usr/bin/env python3
"""HiFi-GAN 単体切り分け（item 4）。同一の postnet mel を固定入力に、PyTorch HiFi-GAN と
CoreML HiFi-GAN を複数の (precision × computeUnit) で動かし、出力波形を比較する。

「mel（HiFi-GAN 入力）が同じでも、実行エンジン次第で HiFi-GAN 出力だけが壊れるか」を、
入力を固定して混入なしで確かめるのが目的。compare_stages.py の device 比較が
「postnet までは正常・HiFi-GAN 出力で破綻」を示したのと同じ結論を、HiFi-GAN だけ取り出して
能動的に再現する。

⚠️ 重要な但し書き:
  CoreML の compute_units=cpuAndNE / all を **Mac（M4 Max）** で指定しても、Mac の ANE 配置は
  iPhone の ANE 配置と一致する保証がない。実機 ANE の破綻（rms ~4 倍）の真値は
  audio/.../debug/<run>/waveform_predeemph.npy にあり、本スクリプトはそれを参照値として
  併記する（--device-good / --device-broken）。Mac ローカル結果は補助・ハーネス検証用。

条件（既定）:
  - PyTorch HiFi-GAN（CPU/MPS）  … 比較の基準
  - CoreML Float32 cpuOnly
  - CoreML Float16 cpuAndGPU
  - CoreML Float16 cpuAndNE
  - CoreML Int8    cpuAndGPU
  - CoreML Int8    cpuAndNE

入力 postnet（HiFi-GAN 入力 mel）は既定で実機 良 run の postnet_output.npy（[262,256]）を使う。
fixed262 モデルは T=262 固定なので、262 フレームの postnet を渡す必要がある。

実行例:
    PronounSE/venv/bin/python scripts/compare_hifigan_isolation.py \\
        --postnet audio/iPhone_3_phase1_20260519/debug/20260519_212806_f16GpuRepeat2_Float16_cpuAndGPU/postnet_output.npy \\
        --md docs/2026-06-21/hifigan-isolation.md --save-wav
"""

from __future__ import annotations

import warnings
warnings.simplefilter("ignore", FutureWarning)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t
import coremltools as ct

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "PronounSE"))
sys.path.insert(0, str(REPO_ROOT / "PronounSE" / "HiFiGAN"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage_metrics as sm  # noqa: E402
from HiFiGAN.models import Generator  # noqa: E402
from HiFiGAN.env import AttrDict  # noqa: E402

HIFIGAN_CHKPT_DIR = REPO_ROOT / "PronounSE/HiFiGAN/chkpt"
MODELS_DIR = REPO_ROOT / "ios/CoreMLAudioApp/CoreMLAudioApp/MLModels"

# (label, model file, compute_unit)
COREML_CONDITIONS = [
    ("F32 cpuOnly", "HiFiGAN_Generator_float32_fixed262.mlpackage", ct.ComputeUnit.CPU_ONLY),
    ("F16 cpuAndGPU", "HiFiGAN_Generator_float16_fixed262.mlpackage", ct.ComputeUnit.CPU_AND_GPU),
    ("F16 cpuAndNE", "HiFiGAN_Generator_float16_fixed262.mlpackage", ct.ComputeUnit.CPU_AND_NE),
    ("Int8 cpuAndGPU", "HiFiGAN_Generator_int8_fixed262.mlpackage", ct.ComputeUnit.CPU_AND_GPU),
    ("Int8 cpuAndNE", "HiFiGAN_Generator_int8_fixed262.mlpackage", ct.ComputeUnit.CPU_AND_NE),
]


def load_postnet(path: Path) -> np.ndarray:
    """postnet を [1, 256, T]（HiFi-GAN 入力形）に整える。"""
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"unexpected postnet shape: {arr.shape}")
    # [T, 256] か [256, T] か判定（n_mels=256）
    if arr.shape[1] == 256:
        arr = arr.T  # [256, T]
    elif arr.shape[0] != 256:
        raise ValueError(f"neither axis is 256: {arr.shape}")
    return arr[np.newaxis, :, :]  # [1, 256, T]


def run_pytorch(vocoder_in: np.ndarray) -> np.ndarray:
    device = t.device("mps" if t.backends.mps.is_available() else "cpu")
    with open(HIFIGAN_CHKPT_DIR / "config.json") as f:
        param = AttrDict(json.load(f))
    t.manual_seed(param.seed)
    g = Generator(param).to(device)
    chkpt = t.load(HIFIGAN_CHKPT_DIR / "g_00009000", map_location=device)
    g.load_state_dict(chkpt["generator"])
    g.eval()
    g.remove_weight_norm()
    x = t.from_numpy(vocoder_in).to(device)
    with t.no_grad():
        y = g(x).squeeze().detach().cpu().numpy().astype(np.float32)
    return y


def run_coreml(model_path: Path, compute_unit, vocoder_in: np.ndarray) -> np.ndarray:
    mdl = ct.models.MLModel(str(model_path), compute_units=compute_unit)
    in_name = list(mdl.input_description)[0]
    out = mdl.predict({in_name: vocoder_in})
    y = np.asarray(list(out.values())[0]).squeeze().astype(np.float32)
    return y


def device_truth_rms(run_dir: Path) -> float | None:
    p = run_dir / "waveform_predeemph.npy"
    if not p.exists():
        return None
    a = np.load(p).astype(np.float64).reshape(-1)
    return float(np.sqrt(np.mean(a * a)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_postnet = (REPO_ROOT / "audio/iPhone_3_phase1_20260519/debug/"
                       "20260519_212806_f16GpuRepeat2_Float16_cpuAndGPU/postnet_output.npy")
    parser.add_argument("--postnet", default=str(default_postnet),
                        help="HiFi-GAN 入力にする postnet_output.npy（既定: 実機 良 run, 262 フレーム）")
    parser.add_argument("--md", help="Markdown 出力先")
    parser.add_argument("--save-wav", action="store_true", help="各条件の出力 wav を保存")
    parser.add_argument("--wav-dir", help="wav 保存先（既定: --md の dir の hifigan_wav/）")
    parser.add_argument("--device-broken", help="実機 ANE 壊れ run dir（真値 rms 併記用）")
    args = parser.parse_args()

    postnet_path = Path(args.postnet)
    if not postnet_path.is_file():
        print(f"# postnet not found: {postnet_path}", file=sys.stderr)
        return 1
    vocoder_in = load_postnet(postnet_path)
    T = vocoder_in.shape[2]
    print(f"# postnet input -> {vocoder_in.shape} (T={T})  from {postnet_path}")

    wav_dir = None
    if args.save_wav:
        if args.wav_dir:
            wav_dir = Path(args.wav_dir)
        elif args.md:
            wav_dir = Path(args.md).parent / "hifigan_wav"
        else:
            wav_dir = REPO_ROOT / "result/hifigan_isolation"
        wav_dir.mkdir(parents=True, exist_ok=True)

    # 基準: PyTorch HiFi-GAN
    print("# running PyTorch HiFi-GAN ...")
    y_ref = run_pytorch(vocoder_in)
    ref_stats = sm.basic_stats(y_ref)
    print(f"  PyTorch  rms={ref_stats['rms']:.5g} peak={ref_stats['peak']:.5g}")

    rows: list[tuple[str, dict]] = []
    for label, model_file, cu in COREML_CONDITIONS:
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            print(f"# model missing, skip: {model_path}", file=sys.stderr)
            continue
        print(f"# running CoreML {label} ({model_file}) ...")
        try:
            y = run_coreml(model_path, cu, vocoder_in)
        except Exception as e:  # pragma: no cover
            print(f"  ! failed: {e}", file=sys.stderr)
            continue
        m = sm.compare(y_ref, y, kind="waveform")
        rows.append((label, m))
        print(f"  {label:16s} cos={m['cosine']:.4f} rms_ratio={m['rms_ratio']:.3f} "
              f"peak={m['peak']:.4g} {m['verdict']}")
        if wav_dir is not None:
            import soundfile as sf
            sf.write(wav_dir / f"hifigan_{label.replace(' ', '_')}.wav", y, 22050, subtype="PCM_16")

    if wav_dir is not None:
        import soundfile as sf
        sf.write(wav_dir / "hifigan_PyTorch.wav", y_ref, 22050, subtype="PCM_16")

    # --- Markdown ---
    lines: list[str] = []
    lines.append(f"HiFi-GAN 入力（postnet）: `{postnet_path}`  shape={vocoder_in.shape}")
    lines.append("")
    lines.append("基準 = PyTorch HiFi-GAN（同じ postnet 入力）。判定は波形なので振幅(rms_ratio)主。")
    lines.append("")
    lines.append("| stage | condition | MAE | RMSE | cosine | min | max | mean | std | peak | rms | NaN/Inf | 判定 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    # PyTorch 基準行（self-compare）
    self_m = sm.compare(y_ref, y_ref, kind="waveform")
    lines.append(sm.md_row("HiFi-GAN出力", "PyTorch（基準）", self_m))
    for label, m in rows:
        lines.append(sm.md_row("HiFi-GAN出力", f"CoreML {label}", m))
    lines.append("")

    # 実機 ANE 真値の rms（Mac ローカルが iPhone ANE を再現できているかの照合）
    if args.device_broken:
        dbroken = Path(args.device_broken)
        rms_b = device_truth_rms(dbroken)
        lines.append("### 実機 ANE 真値との照合")
        lines.append("")
        lines.append(f"- PyTorch HiFi-GAN 出力 rms = {ref_stats['rms']:.5g}")
        if rms_b is not None:
            lines.append(f"- 実機 ANE 壊れ run の HiFi-GAN 出力 rms = {rms_b:.5g} "
                         f"（`{dbroken.name}`）→ PyTorch 比 {rms_b/ref_stats['rms']:.2f}×")
        lines.append("- Mac ローカルの cpuAndNE が上の実機 rms 跳ねを再現できていなければ、"
                     "「Mac の ANE ≠ iPhone の ANE」であり、真値は実機 npy 側に置く。")
        lines.append("")

    md_text = "\n".join(lines) + "\n"
    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md_text)
        print(f"# wrote {out}")
        if wav_dir is not None:
            print(f"# wav -> {wav_dir}")
    else:
        print("\n" + md_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
