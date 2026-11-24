"""開発者が提供したオリジナル PronounSE の合成音を、どの前処理で再現できるかを調べる

公開実装（PronounSE/Transformer/utils.get_spectrograms）は、メルを作る前に
  (1) 先頭の無音を削る（RMS で onset を検出）  (2) 末尾 15% を線形にフェードアウトする
を行う。提供された合成音の長さは (1) を行っていないことを示しているので、次の候補で PyTorch 版を動かし、
提供された合成音（<参照>_<話者>_pronounse.wav）と波形を比べる。
  public       : 公開実装のまま（削る + フェード）
  notrim_fade  : 削らない + フェード
  notrim_nofade: 削らない + フェードしない
メルの計算は get_spectrograms をそのまま写したもの（public が get_spectrograms と完全に一致することを確かめる）。
後処理は synthesis.py と同じ（HiFi-GAN → lfilter によるデエンファシス → soundfile で PCM_16）。

使い方
  PronounSE/venv/bin/python scripts/identify_original_preprocess.py \
      --input s0p5=<口真似音 wav> ... --dataset ~/Desktop --out metrics/preprocess_original_<日付>
"""

import argparse
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PronounSE", "Transformer"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hyperparams as hp  # noqa: E402
from utils import get_spectrograms  # noqa: E402

from convert_hifigan import load_generator  # noqa: E402
from convert_transformer import load_model  # noqa: E402
from verify_variable_length import deemphasis, run_pytorch  # noqa: E402

VARIANTS = {"public": (True, True), "notrim_fade": (False, True), "notrim_nofade": (False, False)}


def onset_index(y, onset_threshold=0.08, shift=256, window_length=1024):
    sq = y ** 2
    for i in range(0, len(sq) - window_length, shift):
        if np.sum(sq[i:i + window_length]) > onset_threshold:
            return i
    return 0


def spectrogram(path, trim, fade, fade_out_ratio=0.15):
    """get_spectrograms と同じ計算。trim / fade だけを切り替えられる"""
    y, _ = librosa.load(path, sr=hp.sr)
    if trim:
        y = y[onset_index(y):]
    if fade:
        n = int(len(y) * fade_out_ratio)
        if n > 0:
            y[len(y) - n:] *= np.linspace(1.0, 0.0, n)
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
    mag = np.abs(librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length))
    mel = np.dot(librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels), mag)
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    return np.ascontiguousarray(mel.T, dtype=np.float32)


def to_pcm16(x):
    """soundfile の PCM_16 書き出しと同じ量子化（±1 で切る）を、ファイルを経由して行う"""
    import io
    buf = io.BytesIO()
    sf.write(buf, x, hp.sr, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return sf.read(buf, dtype="float32")[0]


def logmel_db(x):
    m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=hp.sr, n_fft=1024, hop_length=256, n_mels=80)
    return 10 * np.log10(np.maximum(m, 1e-10))


def compare(ref, x):
    """波形の SNR と、ログメル（80 帯域、dB）の平均絶対差。ログメルは全体と、時間を 4 等分した区間ごと、末尾 15% を出す
    （波形の SNR は位相のずれに敏感なので、スペクトルの近さも見る）"""
    n = min(len(ref), len(x))
    r, y = ref[:n].astype(np.float64), x[:n].astype(np.float64)
    snr = 10 * np.log10(np.sum(r ** 2) / max(np.sum((r - y) ** 2), 1e-300))
    d = np.abs(logmel_db(ref[:n]) - logmel_db(x[:n])).mean(axis=0)  # フレームごと
    q = np.array_split(d, 4)
    tail = d[int(len(d) * 0.85):]
    return {"len_ref": int(len(ref)), "len_x": int(len(x)), "snr_db": float(snr), "max_abs": float(np.abs(r - y).max()),
            "identical_pcm16": bool(len(ref) == len(x) and np.array_equal(ref, x)),
            "logmel_mae_db": float(d.mean()), "logmel_mae_db_quarters": [float(v.mean()) for v in q],
            "logmel_mae_db_last15": float(tail.mean())}


def run_pytorch_on(model, generator, mel, device):
    """run_pytorch と同じ手順を device の上で行う（synthesis.py と同じく Decoder を T 回まわす）"""
    T = mel.shape[0]
    mel_src = torch.from_numpy(mel).unsqueeze(0).to(device)
    pos_src = torch.arange(1, T + 1).unsqueeze(0).to(device)
    dec_in = torch.zeros(1, 1, hp.n_mels, device=device)
    memory, c_mask = model.procEncoder(mel_src, pos_src)
    for _ in range(T):
        pos_trg = torch.arange(1, dec_in.size(1) + 1, device=device).unsqueeze(0)
        mel_pred, postnet = model.procDecoder(memory, dec_in, pos_trg, c_mask)
        dec_in = torch.cat([dec_in, mel_pred[:, -1:, :]], dim=1)
    wave = generator(postnet.transpose(1, 2)).squeeze().cpu().numpy().astype(np.float32)
    return postnet[0].cpu().numpy(), wave


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True, metavar="NAME=WAV")
    ap.add_argument("--dataset", required=True, help="Vocal_Mimicry と PronounSE があるフォルダ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mps-check", action="store_true", help="CPU と MPS の差（計算機の違いだけによる差）も出す")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    model, generator = load_model(), load_generator()

    report = []
    for spec in args.input:
        name, _, path = spec.partition("=")
        spk = os.path.basename(os.path.dirname(path))
        orig_path = os.path.join(args.dataset, "PronounSE", spk,
                                 os.path.basename(path).replace(".wav", "_pronounse.wav"))
        orig, sr = sf.read(orig_path, dtype="float32")
        y22 = librosa.load(path, sr=hp.sr)[0]
        public_exact = np.array_equal(spectrogram(path, True, True), get_spectrograms(path)[0])
        entry = {"name": name, "wav": path, "original": orig_path, "input_samples_22050": int(len(y22)),
                 "onset_index": int(onset_index(y22.copy())), "original_samples": int(len(orig)),
                 "spectrogram_public_equals_get_spectrograms": bool(public_exact), "variants": {}}
        print(f"[{name}] N={len(y22)} onset={entry['onset_index']} オリジナル {len(orig)} サンプル "
              f"(1+floor(N/256))*256={(1 + len(y22) // 256) * 256}  public==get_spectrograms: {public_exact}",
              flush=True)
        waves = {}
        for v, (trim, fade) in VARIANTS.items():
            mel = spectrogram(path, trim, fade)
            np.save(os.path.join(args.out, f"{name}_{v}_mel.npy"), mel)
            _, raw = run_pytorch(model, generator, mel)
            wave = to_pcm16(deemphasis(raw))
            waves[v] = wave
            np.save(os.path.join(args.out, f"{name}_{v}_wave.npy"), wave)
            c = compare(orig, wave)
            c["T"] = int(mel.shape[0])
            entry["variants"][v] = c
            print(f"  {v:<14} T={c['T']:<4} 長さ {c['len_x']} / {c['len_ref']}  SNR {c['snr_db']:5.1f} dB  "
                  f"ログメル差 {c['logmel_mae_db']:.2f} dB（4 区間 {', '.join(f'{q:.2f}' for q in c['logmel_mae_db_quarters'])}"
                  f"、末尾 15% {c['logmel_mae_db_last15']:.2f}）  PCM16 完全一致 {c['identical_pcm16']}", flush=True)
        if args.mps_check and torch.backends.mps.is_available():
            # 同じ前処理（削らない・フェードしない）で、計算機の違いだけによる差の大きさを見る基準
            mel = spectrogram(path, False, False)
            m_mps, g_mps = model.to("mps"), generator.to("mps")
            _, raw = run_pytorch_on(m_mps, g_mps, mel, "mps")
            model.to("cpu"), generator.to("cpu")
            wave = to_pcm16(deemphasis(raw))
            c = compare(waves["notrim_nofade"], wave)
            entry["cpu_vs_mps_notrim_nofade"] = c
            print(f"  [基準] CPU と MPS（notrim_nofade どうし） SNR {c['snr_db']:5.1f} dB  ログメル差 {c['logmel_mae_db']:.2f} dB"
                  f"（4 区間 {', '.join(f'{q:.2f}' for q in c['logmel_mae_db_quarters'])}）", flush=True)
        report.append(entry)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
