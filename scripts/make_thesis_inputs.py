#!/usr/bin/env python3
"""卒論第4章の計測に使う入力と PyTorch 基準を用意する。

1. ASJ152 デモの口真似4本をダウンロードする（data/thesis_ch4/inputs/。gitignore 対象）
2. 各入力を PyTorch 版で Decoder まで通し、出力メル（postnet）を 262 フレームに揃えて保存する
   - 実機と同じく、入力メルを先頭 T' = min(T, 262) フレームに切ってから Encoder に入れ、
     Decoder を T' ステップ回す。T' < 262 の場合、残りのフレームは 0 で埋める
3. 揃えたメルを PyTorch 版 HiFi-GAN に通した波形（HiFi-GAN だけの比較の基準）を保存する。
   実機の VocoderRunner と同じく、出力は先頭 T' * 256 サンプルに切る
4. wav と揃えたメルをアプリの Input/ にコピーする（Input/ も gitignore 対象）

出力:
  data/thesis_ch4/pytorch_reference/<id>/
    mel_normalized.npy        [T, 256]   librosa の前処理後（切る前）
    postnet_output_262.npy    [262, 256] HiFi-GAN に入れるメル
    waveform_predeemph.npy    [1, T'*256]
    waveform_postdeemph.npy   [1, T'*256]
    summary.json
  data/thesis_ch4/pytorch_reference/frames.csv  入力ごとの T と T'
  ios/CoreMLAudioApp/CoreMLAudioApp/Input/thesis_src_<id>.wav
  ios/CoreMLAudioApp/CoreMLAudioApp/Input/thesis_pytorch_postnet_<id>.npy
  ios/CoreMLAudioApp/CoreMLAudioApp/Input/thesis_inputs.json  入力の順番と PyTorch の T / T'

実行例:
    PronounSE/venv/bin/python scripts/make_thesis_inputs.py
"""

from __future__ import annotations

import warnings
warnings.simplefilter("ignore", FutureWarning)

import argparse
import csv
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch as t
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "PronounSE"))
sys.path.insert(0, str(REPO_ROOT / "PronounSE" / "Transformer"))
sys.path.insert(0, str(REPO_ROOT / "PronounSE" / "HiFiGAN"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Transformer.utils import get_spectrograms  # noqa: E402
import hyperparams as hp  # noqa: E402
from extract_pytorch_stages import load_transformer, load_hifigan, stat  # noqa: E402
import extract_pytorch_stages  # noqa: E402

INPUT_IDS = ["1", "6", "11", "2"]
URL_TEMPLATE = "https://www.cc.kyoto-su.ac.jp/~i2486110/asj152/wavs/{id}_src_tk_1.wav"
TARGET_T = 262
HOP = 256

DATA_DIR = REPO_ROOT / "data/thesis_ch4"
APP_INPUT_DIR = REPO_ROOT / "ios/CoreMLAudioApp/CoreMLAudioApp/Input"


def download(input_id: str, dest_dir: Path) -> Path:
    dest = dest_dir / f"{input_id}_src_tk_1.wav"
    if dest.is_file():
        return dest
    url = URL_TEMPLATE.format(id=input_id)
    print(f"# download {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", action="store_true", default=True,
                        help="PyTorch を CPU で動かす（基準なので既定で CPU）")
    args = parser.parse_args()

    # 基準は CPU の float32 で作る（MPS との差を混ぜない）
    if args.cpu:
        extract_pytorch_stages.DEVICE = t.device("cpu")
    device = extract_pytorch_stages.DEVICE

    inputs_dir = DATA_DIR / "inputs"
    ref_root = DATA_DIR / "pytorch_reference"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    ref_root.mkdir(parents=True, exist_ok=True)
    APP_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    m = load_transformer()
    g = load_hifigan()

    rows = []
    for input_id in INPUT_IDS:
        wav_path = download(input_id, inputs_dir)
        out_dir = ref_root / input_id
        out_dir.mkdir(parents=True, exist_ok=True)

        mel_np, _ = get_spectrograms(str(wav_path))  # [T, 256]
        total_t = mel_np.shape[0]
        t_prime = min(total_t, TARGET_T)
        print(f"# input={input_id} T={total_t} T'={t_prime} device={device}")

        summary = {"mel_normalized": stat("mel_normalized", mel_np)}
        np.save(out_dir / "mel_normalized.npy", mel_np)

        mel_src = t.FloatTensor(mel_np[:t_prime]).unsqueeze(0).to(device)
        pos_src = t.LongTensor(np.arange(1, t_prime + 1)).to(device)
        with t.no_grad():
            memory, c_mask = m.procEncoder(mel_src, pos_src)
            mel_trg_input = t.zeros([1, 1, hp.n_mels]).to(device)
            for _ in range(t_prime):
                pos_trg = t.arange(1, mel_trg_input.size(1) + 1).unsqueeze(0).to(device)
                mel_pred, postnet_pred = m.procDecoder(memory, mel_trg_input, pos_trg, c_mask)
                mel_trg_input = t.cat([mel_trg_input, mel_pred[:, -1:, :]], dim=1)
        postnet = postnet_pred.squeeze(0).cpu().numpy().astype(np.float32)  # [T', 256]

        postnet_262 = np.zeros((TARGET_T, hp.n_mels), dtype=np.float32)
        postnet_262[:t_prime] = postnet[:t_prime]
        summary["postnet_output_262"] = stat("postnet_output_262", postnet_262)
        np.save(out_dir / "postnet_output_262.npy", postnet_262)

        with t.no_grad():
            vocoder_in = t.FloatTensor(postnet_262.T).unsqueeze(0).to(device)  # [1, 256, 262]
            y_full = g(vocoder_in).squeeze().cpu().numpy().astype(np.float32)
        y_pre = y_full[: t_prime * HOP]
        y_post = signal.lfilter([1], [1, -hp.preemphasis], y_pre).astype(np.float32)
        summary["waveform_predeemph"] = stat("waveform_predeemph", y_pre[np.newaxis, :])
        summary["waveform_postdeemph"] = stat("waveform_postdeemph", y_post[np.newaxis, :])
        np.save(out_dir / "waveform_predeemph.npy", y_pre[np.newaxis, :])
        np.save(out_dir / "waveform_postdeemph.npy", y_post[np.newaxis, :])

        meta = {"input": str(wav_path), "url": URL_TEMPLATE.format(id=input_id),
                "device": str(device), "T": int(total_t), "T_prime": int(t_prime),
                "sr": hp.sr, "stages": summary}
        (out_dir / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        shutil.copyfile(wav_path, APP_INPUT_DIR / f"thesis_src_{input_id}.wav")
        np.save(APP_INPUT_DIR / f"thesis_pytorch_postnet_{input_id}.npy", postnet_262)
        rows.append({"input_name": input_id, "T": total_t, "T_prime": t_prime})

    with open(ref_root / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input_name", "T", "T_prime"])
        w.writeheader()
        w.writerows(rows)
    (APP_INPUT_DIR / "thesis_inputs.json").write_text(json.dumps({"inputs": rows}, indent=2))
    print(f"# saved references to {ref_root}")
    print(f"# copied inputs to {APP_INPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
