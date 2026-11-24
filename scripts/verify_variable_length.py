"""可変長（RangeDim）の Core ML モデルが、入力の全フレームを PyTorch と同じように処理できるかを確かめる

同じ入力 wav から PronounSE の前処理（Transformer/utils.get_spectrograms）でメルを 1 回だけ作り、
PyTorch 版と Core ML 版（coremltools で Mac 上で実行）の両方にそのメルを入れる。
どちらも synthesis.py と同じく Decoder を T 回まわし（入力は 0 の 1 フレームから始めて 1 フレームずつ伸ばす）、
HiFi-GAN の後にデエンファシス（lfilter）をかける。

確かめること（入力ごと）
  - 入力が途中で切られない（Core ML の Decoder の出力が T フレーム）
  - 出力のサンプル数が T × 256
  - NaN・Inf が無い
  - クリッピングが無い（デエンファシス後の |x| の最大が 1 未満。16 ビットの wav に保存すると ±1 で切られる）
  - PyTorch との差（postnet のメル、HiFi-GAN の出力、デエンファシス後の波形）

出力（--out のフォルダ）
  <name>_mel_in.npy                 前処理後のメル [T × 256]（iOS の実機テストにも同じものを入れる）
  <name>_{pytorch,coreml}_postnet.npy
  <name>_{pytorch,coreml}_wave.npy  デエンファシス後の波形
  summary.json / summary.csv

使い方
  PronounSE/venv/bin/python scripts/verify_variable_length.py \
      --input short=<wav> --input mid=<wav> ... --shape-mode range1 --precision float32 \
      --compute-units CPU_ONLY --out metrics/variable_length_<日付>
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import signal

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PronounSE", "Transformer"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PronounSE", "HiFiGAN"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coremltools as ct  # noqa: E402
import hyperparams as hp  # noqa: E402
from utils import get_spectrograms  # noqa: E402

from convert_hifigan import load_generator  # noqa: E402
from convert_transformer import load_model  # noqa: E402

MODEL_DIR = os.path.join(PROJECT_ROOT, "ios", "CoreMLAudioApp", "CoreMLAudioApp", "MLModels")
HOP = 256


def deemphasis(x):
    return signal.lfilter([1], [1, -hp.preemphasis], x).astype(np.float32)


def run_pytorch(model, generator, mel):
    T = mel.shape[0]
    mel_src = torch.from_numpy(mel).unsqueeze(0)
    pos_src = torch.arange(1, T + 1).unsqueeze(0)
    dec_in = torch.zeros(1, 1, hp.n_mels)
    with torch.no_grad():
        memory, c_mask = model.procEncoder(mel_src, pos_src)
        for _ in range(T):
            pos_trg = torch.arange(1, dec_in.size(1) + 1).unsqueeze(0)
            mel_pred, postnet = model.procDecoder(memory, dec_in, pos_trg, c_mask)
            dec_in = torch.cat([dec_in, mel_pred[:, -1:, :]], dim=1)
        wave = generator(postnet.transpose(1, 2)).squeeze().numpy().astype(np.float32)
    return postnet[0].numpy(), wave


def run_coreml(enc, dec, hifi, mel):
    T = mel.shape[0]
    pos = np.arange(1, T + 1, dtype=np.int32)[None]
    t0 = time.perf_counter()
    memory = list(enc.predict({"mel": mel[None].astype(np.float32), "pos": pos}).values())[0]
    t_enc = time.perf_counter() - t0
    dec_in = np.zeros((1, 1, hp.n_mels), dtype=np.float32)
    postnet = None
    t0 = time.perf_counter()
    for _ in range(T):
        L = dec_in.shape[1]
        out = dec.predict({"memory": memory, "decoder_input": dec_in,
                           "pos": np.arange(1, L + 1, dtype=np.int32)[None]})
        mel_out = out["mel_out"]
        postnet = out[next(k for k in out if k != "mel_out")]
        dec_in = np.concatenate([dec_in, mel_out[:, -1:, :]], axis=1)
    t_dec = time.perf_counter() - t0
    t0 = time.perf_counter()
    wave = list(hifi.predict({"mel": np.ascontiguousarray(postnet.transpose(0, 2, 1))}).values())[0]
    t_hifi = time.perf_counter() - t0
    return postnet[0], np.asarray(wave).reshape(-1).astype(np.float32), memory.shape, (t_enc, t_dec, t_hifi)


def diff(a, b):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    err = np.abs(a64 - b64)
    snr = 10 * np.log10(np.sum(a64 ** 2) / max(np.sum((a64 - b64) ** 2), 1e-300))
    return {"max_abs": float(err.max()), "mean_abs": float(err.mean()), "snr_db": float(snr),
            "cosine": float(a64.ravel() @ b64.ravel() / (np.linalg.norm(a64) * np.linalg.norm(b64)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True, metavar="NAME=WAV")
    ap.add_argument("--shape-mode", default="range1")
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--compute-units", default="CPU_ONLY", choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"])
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preprocess", choices=["public", "full"], default="public",
                    help="PyTorch 側のメルの前処理。public は get_spectrograms のまま（先頭の無音を削り、フェードする）、"
                         "full はオリジナルの合成音を作った条件（削らず、フェードしない）")
    ap.add_argument("--coreml-mel-dir",
                    help="Core ML 側には、このフォルダの <name>_fullInput_mel.npy（iOS の前処理で作ったメル）を入れる。"
                         "省略時は PyTorch 側と同じメル")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cu = getattr(ct.ComputeUnit, args.compute_units)
    names = {k: f"{k}_{args.precision}_{args.shape_mode}.mlpackage"
             for k in ["Transformer_Encoder", "Transformer_Decoder", "HiFiGAN_Generator"]}
    enc, dec, hifi = (ct.models.MLModel(os.path.join(args.model_dir, names[k]), compute_units=cu)
                      for k in ["Transformer_Encoder", "Transformer_Decoder", "HiFiGAN_Generator"])
    model = load_model()
    generator = load_generator()

    rows = []
    for spec in args.input:
        name, _, path = spec.partition("=")
        if args.preprocess == "full":
            # identify_original_preprocess はこのファイルを import するので、ここで import する
            from identify_original_preprocess import spectrogram
            mel = spectrogram(path, trim=False, fade=False)
        else:
            mel, _ = get_spectrograms(path)
        # get_spectrograms は転置した配列を返すので、行優先（iOS の MinimalNpyReader が読める形）にする
        mel = np.ascontiguousarray(mel, dtype=np.float32)
        T = mel.shape[0]
        print(f"[{name}] T={T}（{(T - 1) * HOP / hp.sr:.2f} 秒相当） {path}", flush=True)
        np.save(os.path.join(args.out, f"{name}_mel_in.npy"), mel)
        cm_mel = mel
        if args.coreml_mel_dir:
            cm_mel = np.ascontiguousarray(np.load(os.path.join(args.coreml_mel_dir, f"{name}_fullInput_mel.npy")),
                                          dtype=np.float32)
            if cm_mel.shape != mel.shape:
                raise SystemExit(f"{name}: iOS のメル {cm_mel.shape} と PyTorch のメル {mel.shape} の形が違う")

        pt_post, pt_raw = run_pytorch(model, generator, mel)
        cm_post, cm_raw, mem_shape, times = run_coreml(enc, dec, hifi, cm_mel)
        pt_wave, cm_wave = deemphasis(pt_raw), deemphasis(cm_raw)
        np.save(os.path.join(args.out, f"{name}_pytorch_postnet.npy"), pt_post)
        np.save(os.path.join(args.out, f"{name}_coreml_postnet.npy"), cm_post)
        np.save(os.path.join(args.out, f"{name}_pytorch_wave.npy"), pt_wave)
        np.save(os.path.join(args.out, f"{name}_coreml_wave.npy"), cm_wave)

        row = {
            "name": name, "wav": path, "T": T, "memory_T": int(mem_shape[1]),
            "coreml_postnet_T": int(cm_post.shape[0]), "pytorch_postnet_T": int(pt_post.shape[0]),
            "expected_samples": T * HOP, "coreml_samples": int(cm_wave.size), "pytorch_samples": int(pt_wave.size),
            "coreml_nonfinite": int((~np.isfinite(cm_raw)).sum() + (~np.isfinite(cm_post)).sum()),
            "coreml_peak": float(np.max(np.abs(cm_wave))), "pytorch_peak": float(np.max(np.abs(pt_wave))),
            "coreml_clipped_samples": int((np.abs(cm_wave) >= 1.0).sum()),
            "pytorch_clipped_samples": int((np.abs(pt_wave) >= 1.0).sum()),
            "postnet": diff(pt_post, cm_post), "hifigan_raw": diff(pt_raw, cm_raw), "wave": diff(pt_wave, cm_wave),
            "coreml_sec": {"encoder": times[0], "decoder": times[1], "hifigan": times[2]},
        }
        row["no_truncation"] = row["memory_T"] == T and row["coreml_postnet_T"] == T
        row["length_ok"] = row["coreml_samples"] == row["expected_samples"] == row["pytorch_samples"]
        rows.append(row)
        print(f"  切り詰めなし={row['no_truncation']} 長さ={row['coreml_samples']}（期待 {T * HOP}） "
              f"NaN/Inf={row['coreml_nonfinite']} peak={row['coreml_peak']:.3f}（PyTorch {row['pytorch_peak']:.3f}） "
              f"postnet max|差|={row['postnet']['max_abs']:.2e} 波形 SNR={row['wave']['snr_db']:.1f} dB "
              f"Decoder {times[1]:.1f} 秒", flush=True)

    meta = {"preprocess": args.preprocess, "coreml_mel_dir": args.coreml_mel_dir,
            "shape_mode": args.shape_mode, "precision": args.precision, "compute_units": args.compute_units,
            "models": names, "coremltools": ct.__version__, "torch": torch.__version__}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "T", "no_truncation", "length_ok", "samples", "nonfinite", "coreml_peak", "pytorch_peak",
                    "clipped", "postnet_max_abs", "postnet_snr_db", "wave_max_abs", "wave_snr_db", "wave_cosine"])
        for r in rows:
            w.writerow([r["name"], r["T"], r["no_truncation"], r["length_ok"], r["coreml_samples"], r["coreml_nonfinite"],
                        f"{r['coreml_peak']:.4f}", f"{r['pytorch_peak']:.4f}", r["coreml_clipped_samples"],
                        f"{r['postnet']['max_abs']:.3e}", f"{r['postnet']['snr_db']:.1f}",
                        f"{r['wave']['max_abs']:.3e}", f"{r['wave']['snr_db']:.1f}", f"{r['wave']['cosine']:.6f}"])
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
