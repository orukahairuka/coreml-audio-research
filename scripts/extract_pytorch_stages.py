#!/usr/bin/env python3
"""PyTorch baseline の合成パイプラインを 1 本走らせ、各ステージ境界の中間出力を保存する。

ANE 経路で音が壊れる箇所を切り分けるための「真値（reference）」を作るスクリプト。
`PronounSE/synthesis.py` と同じ処理を再現しつつ、5 つの境界を `.npy`（波形は `.wav` も）で書き出す。
既存の synthesis.py / synthesis_coreml.py は一切変更しない。

保存する境界（iOS の debug snapshot と同じファイル名に揃えてある）:
  - mel_normalized.npy     : encoder 入力メルスペクトログラム（get_spectrograms 出力, [T, 256]）
  - encoder_output.npy     : Encoder 出力 memory（[T, 512]）
  - postnet_output.npy     : Decoder/postnet 出力 = HiFi-GAN 入力 mel（[T, 256]）
  - waveform_predeemph.npy : HiFi-GAN 出力波形（de-emphasis 前, [1, N]）
  - waveform_postdeemph.npy: 最終波形（de-emphasis 後, [1, N]）
  - waveform_predeemph.wav / waveform_postdeemph.wav
  - summary.json           : 各境界の統計

iOS の debug snapshot（audio/<archive>/debug/<runId>/）は同じファイル名で同じ境界を
保存しているので、compare_stages.py でそのまま段階比較できる。

実行例:
    cd PronounSE && venv/bin/python ../scripts/extract_pytorch_stages.py \\
        --input input_sample.wav \\
        --out-dir ../data/2026-06-21/pytorch_reference/input_sample
"""

from __future__ import annotations

import warnings
warnings.simplefilter("ignore", FutureWarning)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch as t
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "PronounSE"))
sys.path.insert(0, str(REPO_ROOT / "PronounSE" / "Transformer"))
sys.path.insert(0, str(REPO_ROOT / "PronounSE" / "HiFiGAN"))

from Transformer.utils import get_spectrograms  # noqa: E402
from Transformer.network import Model  # noqa: E402
import hyperparams as hp  # noqa: E402
from HiFiGAN.models import Generator  # noqa: E402
from HiFiGAN.env import AttrDict  # noqa: E402

DEVICE = t.device("mps" if t.backends.mps.is_available() else "cpu")
TRANSFORMER_CHKPT = REPO_ROOT / "PronounSE/Transformer/chkpt/chkpt__20000.pth.tar"
HIFIGAN_CHKPT_DIR = REPO_ROOT / "PronounSE/HiFiGAN/chkpt"


def load_transformer() -> Model:
    m = Model(hp.prenet_type).to(DEVICE)
    chkpt = t.load(TRANSFORMER_CHKPT, map_location=DEVICE)
    m.load_state_dict(chkpt["model"])
    m.train(False)
    return m


def load_hifigan() -> Generator:
    with open(HIFIGAN_CHKPT_DIR / "config.json") as f:
        param = AttrDict(json.load(f))
    t.manual_seed(param.seed)
    g = Generator(param).to(DEVICE)
    chkpt = t.load(HIFIGAN_CHKPT_DIR / "g_00009000", map_location=DEVICE)
    g.load_state_dict(chkpt["generator"])
    g.eval()
    g.remove_weight_norm()
    return g


def stat(name: str, arr: np.ndarray) -> dict:
    flat = arr.astype(np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(flat * flat)))
    s = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": rms,
        "has_nan": bool(np.isnan(flat).any()),
        "has_inf": bool(np.isinf(flat).any()),
    }
    print(f"  {name:22s} shape={str(arr.shape):14s} min={s['min']:.5g} "
          f"max={s['max']:.5g} mean={s['mean']:.5g} rms={rms:.5g}")
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(REPO_ROOT / "PronounSE/input_sample.wav"),
                        help="入力 wav（既定: PronounSE/input_sample.wav）")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data/2026-06-21/pytorch_reference/input_sample"))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"# input not found: {input_path}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# device={DEVICE}  input={input_path}")

    # --- mel（encoder 入力） ---
    mel_np, _ = get_spectrograms(str(input_path))  # [T, 256], 正規化済み
    pos = np.arange(1, mel_np.shape[0] + 1)
    mel_src = t.FloatTensor(mel_np).unsqueeze(0).to(DEVICE)   # [1, T, 256]
    pos_src = t.LongTensor(pos).to(DEVICE)
    T_src = mel_np.shape[0]
    print(f"# T(frames)={T_src}")

    m = load_transformer()
    g = load_hifigan()

    summary: dict = {}
    summary["mel_normalized"] = stat("mel_normalized", mel_np)
    np.save(out_dir / "mel_normalized.npy", mel_np)

    # --- Encoder ---
    with t.no_grad():
        memory, c_mask = m.procEncoder(mel_src, pos_src)   # memory [1, T, 512]
    encoder_out = memory.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [T, 512]
    summary["encoder_output"] = stat("encoder_output", encoder_out)
    np.save(out_dir / "encoder_output.npy", encoder_out)

    # --- Decoder（自己回帰ループ, synthesis.py と同じ） ---
    mel_trg_input = t.zeros([1, 1, hp.n_mels]).to(DEVICE)
    with t.no_grad():
        for _ in range(T_src):
            pos_trg = t.arange(1, mel_trg_input.size(1) + 1).unsqueeze(0).to(DEVICE)
            mel_pred, postnet_pred = m.procDecoder(memory, mel_trg_input, pos_trg, c_mask)
            mel_trg_input = t.cat([mel_trg_input, mel_pred[:, -1:, :]], dim=1)
        # postnet_pred: [1, T, 256]（HiFi-GAN へ渡す前）
        postnet_out = postnet_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [T, 256]
    summary["postnet_output"] = stat("postnet_output", postnet_out)
    np.save(out_dir / "postnet_output.npy", postnet_out)

    # --- HiFi-GAN ---
    with t.no_grad():
        vocoder_in = t.transpose(postnet_pred, 1, 2)  # [1, 256, T]
        y_pre = g(vocoder_in).squeeze().to("cpu").detach().numpy().copy().astype(np.float32)
    waveform_predeemph = y_pre[np.newaxis, :]  # [1, N]
    summary["waveform_predeemph"] = stat("waveform_predeemph", waveform_predeemph)
    np.save(out_dir / "waveform_predeemph.npy", waveform_predeemph)

    # --- de-emphasis（最終波形） ---
    y_post = signal.lfilter([1], [1, -hp.preemphasis], y_pre).astype(np.float32)
    waveform_postdeemph = y_post[np.newaxis, :]  # [1, N]
    summary["waveform_postdeemph"] = stat("waveform_postdeemph", waveform_postdeemph)
    np.save(out_dir / "waveform_postdeemph.npy", waveform_postdeemph)

    # wav も保存（聴感確認用）
    sf.write(out_dir / "waveform_predeemph.wav", y_pre, hp.sr, subtype="PCM_16")
    sf.write(out_dir / "waveform_postdeemph.wav", y_post, hp.sr, subtype="PCM_16")

    meta = {"input": str(input_path), "device": str(DEVICE), "T_frames": int(T_src),
            "sr": hp.sr, "stages": summary}
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"# saved reference to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
