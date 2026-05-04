"""HiFi-GAN を PyTorch / Core ML CPU / Core ML GPU の3経路で実行して比較する

入力 wav から PyTorch Encoder + Decoder で postnet_out を一度作り、それを
3つの HiFi-GAN 実装に同じ入力として渡して出力波形の統計を比較する。

使い方:
    PronounSE/venv/bin/python scripts/compare_hifigan_paths.py [input.wav]
"""

import warnings
warnings.simplefilter("ignore", FutureWarning)

import os
import sys
import json

import numpy as np
import torch
import coremltools as ct

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PronounSE"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PronounSE", "Transformer"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PronounSE", "HiFiGAN"))

from Transformer.utils import get_spectrograms
from Transformer.network import Model
import Transformer.hyperparams as hp
from HiFiGAN.models import Generator
from HiFiGAN.env import AttrDict

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TRANSFORMER_CHKPT = os.path.join(PROJECT_ROOT, "PronounSE", "Transformer", "chkpt", "chkpt__20000.pth.tar")
HIFIGAN_CHKPT_DIR = os.path.join(PROJECT_ROOT, "PronounSE", "HiFiGAN", "chkpt")
COREML_HIFIGAN = os.path.join(
    PROJECT_ROOT,
    "ios", "CoreMLAudioApp", "CoreMLAudioApp", "MLModels",
    "HiFiGAN_Generator_float32.mlpackage",
)


def stats(name, arr):
    a = np.asarray(arr).reshape(-1).astype(np.float64)
    print(
        f"  {name:24s} shape={np.asarray(arr).shape}  "
        f"min={a.min(): .6e}  max={a.max(): .6e}  mean={a.mean(): .6e}  "
        f"hasNaN={np.isnan(a).any()}  hasInf={np.isinf(a).any()}"
    )
    print(f"    head[0:5]={a[:5]}  tail[-5:]={a[-5:]}")


def get_postnet_out(input_wav):
    """PyTorch Transformer で input_wav -> postnet_out [1, 256, T] を作る"""
    print(f"PyTorch Transformer を実行: {input_wav}")
    mel, _ = get_spectrograms(input_wav)
    pos = np.arange(1, mel.shape[0] + 1)
    mel_t = torch.FloatTensor(mel).unsqueeze(0).to(DEVICE)
    pos_t = torch.LongTensor(pos).to(DEVICE)

    m = Model(hp.prenet_type).to(DEVICE)
    chkpt = torch.load(TRANSFORMER_CHKPT, map_location=DEVICE)
    m.load_state_dict(chkpt["model"])
    m.eval()

    mel_trg_input = torch.zeros([1, 1, hp.n_mels]).to(DEVICE)
    with torch.no_grad():
        memory, c_mask = m.procEncoder(mel_t, pos_t)
        for _ in range(mel_t.shape[1]):
            pos_trg = torch.arange(1, mel_trg_input.size(1) + 1).unsqueeze(0).to(DEVICE)
            mel_pred, postnet_pred = m.procDecoder(memory, mel_trg_input, pos_trg, c_mask)
            mel_trg_input = torch.cat([mel_trg_input, mel_pred[:, -1:, :]], dim=1)
        postnet_pred = torch.transpose(postnet_pred, 1, 2)  # [1, 256, T]
    return postnet_pred.detach().cpu().numpy().astype(np.float32)


def run_pytorch_hifigan(postnet_out_np):
    print("\n[PyTorch HiFi-GAN] (DEVICE={})".format(DEVICE))
    with open(os.path.join(HIFIGAN_CHKPT_DIR, "config.json")) as f:
        config = AttrDict(json.load(f))
    torch.manual_seed(config.seed)
    g = Generator(config).to(DEVICE)
    chkpt = torch.load(os.path.join(HIFIGAN_CHKPT_DIR, "g_00009000"), map_location=DEVICE)
    g.load_state_dict(chkpt["generator"])
    g.eval()
    g.remove_weight_norm()

    x = torch.from_numpy(postnet_out_np).to(DEVICE)
    with torch.no_grad():
        y = g(x).squeeze().detach().cpu().numpy().astype(np.float32)
    stats("HiFi-GAN out", y)
    return y


def run_coreml_hifigan(postnet_out_np, compute_unit, label):
    print(f"\n[Core ML HiFi-GAN] compute_unit={label}")
    mlmodel = ct.models.MLModel(COREML_HIFIGAN, compute_units=compute_unit)
    out = mlmodel.predict({"mel": postnet_out_np})
    key = list(out.keys())[0]
    y = np.asarray(out[key]).squeeze().astype(np.float32)
    stats(f"HiFi-GAN out ({label})", y)
    return y


def main():
    input_wav = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        PROJECT_ROOT, "PronounSE", "input_sample.wav"
    )
    if not os.path.isfile(input_wav):
        print(f"error: file not found: {input_wav}")
        sys.exit(1)

    cache = os.path.join(PROJECT_ROOT, "result", "postnet_out_cache.npy")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.isfile(cache):
        postnet_out = np.load(cache)
        print(f"キャッシュから読み込み: {cache}  shape={postnet_out.shape}")
    else:
        postnet_out = get_postnet_out(input_wav)
        np.save(cache, postnet_out)
        print(f"キャッシュ保存: {cache}")

    print("\n=== postnet_out (HiFi-GAN 入力) ===")
    stats("postnet_out", postnet_out)

    y_pt = run_pytorch_hifigan(postnet_out)
    y_cpu = run_coreml_hifigan(postnet_out, ct.ComputeUnit.CPU_ONLY, "CPU_ONLY")
    y_gpu = run_coreml_hifigan(postnet_out, ct.ComputeUnit.CPU_AND_GPU, "CPU_AND_GPU")
    y_all = run_coreml_hifigan(postnet_out, ct.ComputeUnit.ALL, "ALL (CPU+GPU+ANE)")

    print("\n=== ペアごとの最大絶対誤差 ===")
    def maxabs(a, b):
        n = min(len(a), len(b))
        return float(np.max(np.abs(a[:n] - b[:n])))
    print(f"  |PyTorch - CoreML CPU|  max = {maxabs(y_pt, y_cpu): .6e}")
    print(f"  |PyTorch - CoreML GPU|  max = {maxabs(y_pt, y_gpu): .6e}")
    print(f"  |PyTorch - CoreML ALL|  max = {maxabs(y_pt, y_all): .6e}")
    print(f"  |CPU     - GPU       |  max = {maxabs(y_cpu, y_gpu): .6e}")
    print(f"  |CPU     - ALL       |  max = {maxabs(y_cpu, y_all): .6e}")


if __name__ == "__main__":
    main()
