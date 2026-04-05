"""CoreML モデルで音声合成パイプラインを実行するスクリプト

PyTorch 版 (PronounSE/synthesis.py) と同じ処理を CoreML モデルで再現し、
出力音声を比較できるようにする。

使い方:
    venv/bin/python scripts/synthesis_coreml.py <input.wav>
    venv/bin/python scripts/synthesis_coreml.py --precision int8 <input.wav>
"""

import argparse
import sys
import os

import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm
import coremltools as ct

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.pardir)

# PronounSE のモジュールを読み込むためにパスを通す
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PronounSE", "Transformer"))
from utils import get_spectrograms
import hyperparams as hp


def get_model_paths(precision):
    return (
        os.path.join(PROJECT_ROOT, f"Transformer_Encoder_{precision}.mlpackage"),
        os.path.join(PROJECT_ROOT, f"Transformer_Decoder_{precision}.mlpackage"),
        os.path.join(PROJECT_ROOT, f"HiFiGAN_Generator_{precision}.mlpackage"),
    )

SAVE_DIR = os.path.join(PROJECT_ROOT, "result")
os.makedirs(SAVE_DIR, exist_ok=True)


def load_coreml_models(precision):
    """指定精度の CoreML モデル 3 つをロードして返す"""
    encoder_path, decoder_path, hifigan_path = get_model_paths(precision)
    print(f"CoreML モデルをロード中... (精度: {precision})")
    encoder = ct.models.MLModel(encoder_path)
    decoder = ct.models.MLModel(decoder_path)
    hifigan = ct.models.MLModel(hifigan_path)
    return encoder, decoder, hifigan


def wav2feature(input_path):
    """入力音声をメルスペクトログラムと位置インデックスに変換する"""
    mel, _ = get_spectrograms(input_path)
    pos = np.arange(1, mel.shape[0] + 1, dtype=np.int32)
    mel = mel[np.newaxis, :, :]  # [1, T, 256]
    pos = pos[np.newaxis, :]     # [1, T]
    return mel, pos


def synthesize(encoder, decoder, hifigan, mel_src, pos_src):
    """CoreML モデルで合成パイプラインを実行する

    処理の流れ:
    1. Encoder: 入力メルスペクトログラム → memory (中間表現)
    2. Decoder: memory を参照しながら自己回帰ループで出力メルスペクトログラムを生成
    3. HiFi-GAN: メルスペクトログラム → 音声波形
    """
    MAX_FRAMES = 1000  # CoreML 変換時の RangeDim 上限
    T_src = mel_src.shape[1]
    if T_src > MAX_FRAMES:
        raise ValueError(f"入力が {T_src} フレームあり、上限 {MAX_FRAMES} フレームを超えています")

    # 1. Encoder
    print("Encoder 実行中...")
    enc_out = encoder.predict({"mel": mel_src, "pos": pos_src})
    memory = list(enc_out.values())[0]  # [1, T_src, 512]

    # 2. Decoder (自己回帰ループ)
    print("Decoder 実行中...")
    mel_trg_input = np.zeros([1, 1, hp.n_mels], dtype=np.float32)

    for _ in tqdm(range(T_src)):
        t_trg = mel_trg_input.shape[1]
        pos_trg = np.arange(1, t_trg + 1, dtype=np.int32)[np.newaxis, :]

        dec_out = decoder.predict({
            "memory": memory,
            "decoder_input": mel_trg_input,
            "pos": pos_trg,
        })

        # Decoder の出力キーから mel_out / postnet_out を取得
        if "mel_out" in dec_out:
            mel_pred = dec_out["mel_out"]
            postnet_pred = [v for k, v in dec_out.items() if k != "mel_out"][0]
        else:
            # 自動リネームされた場合は宣言順（挿入順）で取得
            outputs = list(dec_out.values())
            mel_pred = outputs[0]         # mel_out [1, T_trg, 256]
            postnet_pred = outputs[1]     # postnet_out [1, T_trg, 256]

        last_frame = mel_pred[:, -1:, :]  # [1, 1, 256]
        mel_trg_input = np.concatenate([mel_trg_input, last_frame], axis=1)

    # 3. HiFi-GAN
    print("HiFi-GAN 実行中...")
    # postnet_pred: [1, T, 256] → [1, 256, T] に転置
    vocoder_input = np.transpose(postnet_pred, (0, 2, 1)).astype(np.float32)
    hifi_out = hifigan.predict({"mel": vocoder_input})
    y_hat = list(hifi_out.values())[0]
    y_hat = y_hat.squeeze().astype(np.float32)

    # デエンファシスフィルタ（preemphasis の逆処理）
    y_hat = signal.lfilter([1], [1, -hp.preemphasis], y_hat)

    return y_hat.astype(np.float32)


def main(file_path, precision):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(SAVE_DIR, f"{base_name}_coreml_{precision}.wav")

    encoder, decoder, hifigan = load_coreml_models(precision)
    mel_src, pos_src = wav2feature(file_path)
    print(f"入力: {file_path} ({mel_src.shape[1]} フレーム)")

    y_hat = synthesize(encoder, decoder, hifigan, mel_src, pos_src)

    sf.write(output_path, y_hat, hp.sr, subtype="PCM_16")
    print(f"保存完了: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        choices=["float16", "float32", "int8"],
        default="float16",
    )
    parser.add_argument("input", nargs="?")
    args = parser.parse_args()

    default_input = os.path.join(PROJECT_ROOT, "PronounSE", "input_sample.wav")
    file_path = args.input if args.input else default_input
    if not os.path.isfile(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        sys.exit(1)
    main(file_path, args.precision)
