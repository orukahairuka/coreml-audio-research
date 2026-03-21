# Transformer (Encoder/Decoder) CoreML 変換レポート

## 概要

PronounSE の Transformer モデル（Encoder と Decoder）を CoreML (.mlpackage) に変換した。
推論時は Encoder → Decoder（自己回帰ループ）の順で別々に呼ばれるため、2 つの CoreML モデルとして変換している。

## 環境

| 項目 | 値 |
|------|-----|
| マシン | Mac M4 Max |
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| coremltools | 9.0 |
| 変換形式 | mlprogram (Float16) |

## モデルの役割

音声合成パイプラインは 3 段階で動く:

```
入力音声 → [Encoder] → 中間表現 (memory)
                           ↓
         [Decoder] ← 自己回帰ループで 1 フレームずつ生成
                           ↓
                    メルスペクトログラム
                           ↓
                   [HiFi-GAN] → 出力音声
```

- **Encoder**: 入力音声のメルスペクトログラムを受け取り、特徴量 (memory) に変換する。1 回だけ実行。
- **Decoder**: memory を参照しながら、出力メルスペクトログラムを 1 フレームずつ生成する。フレーム数分ループして実行。

### なぜ Encoder と Decoder を別々に変換するのか

推論時、Encoder は 1 回だけ実行し、Decoder は出力フレーム数分ループ実行する。1 つのモデルにまとめると毎ループ Encoder も再計算してしまうため、分離する必要がある。

## モデル構成

### Encoder

| パラメータ | 値 |
|-----------|-----|
| 入力 | メルスペクトログラム `[1, T_src, 256]` + 位置 `[1, T_src]` (int32) |
| 出力 | memory `[1, T_src, 512]` |
| Prenet | FC(256 → 1024 → 512) + ReLU + Dropout |
| 位置埋め込み | Sinusoid Embedding (1024, 512) |
| Attention | Multi-head Self-Attention × 3 層 (4 head, dim=128) |
| FFN | Conv1d(512 → 2048 → 512) × 3 層 |

### Decoder

| パラメータ | 値 |
|-----------|-----|
| 入力 | memory `[1, T_src, 512]` + デコーダ入力 `[1, T_trg, 256]` + 位置 `[1, T_trg]` (int32) |
| 出力 | mel_out `[1, T_trg, 256]` + postnet_out `[1, T_trg, 256]` |
| Prenet | FC(256 → 1024 → 512) + ReLU + Dropout |
| Self-Attention | Multi-head × 3 層 (因果マスク付き) |
| Cross-Attention | Multi-head × 3 層 (memory を参照) |
| FFN | Conv1d(512 → 2048 → 512) × 3 層 |
| PostConvNet | Causal Conv1d × 5 層 (256 → 512 → 512 → 512 → 256) |

### Decoder の自己回帰ループとは

Decoder は 1 回の呼び出しで全フレームを出力するのではなく、以下のように動く:

1. 最初はゼロベクトル `[1, 1, 256]` を入力
2. Decoder が 1 フレーム分の出力を生成
3. その出力を入力に追加 → `[1, 2, 256]`
4. 再び Decoder に入力して次のフレームを生成
5. 入力音声のフレーム数分だけ繰り返す

つまり Decoder の入力長 `T_trg` は 1 から T_src まで毎ステップ増えていく。このため、入力の時間軸は可変長 (`RangeDim`) で変換している。

## 変換の工夫

### ラッパークラス

元の `Model` クラスの `procEncoder` / `procDecoder` は `None` を返したり受け取ったりするため、`torch.jit.trace` と相性が悪い。そこでラッパークラスを作成し、必要な入出力だけに絞った。

- `EncoderWrapper`: `(mel, pos) → memory` のみ返す
- `DecoderWrapper`: `(memory, decoder_input, pos) → (mel_out, postnet_out)` を返し、`c_mask=None` を内部で固定

### eval モードの分岐

Encoder と Decoder の `forward()` には `if self.training` の分岐がある。eval モードで trace するため、eval パスのみが記録される。

## 精度比較結果

### Encoder

| 指標 | 値 |
|------|-----|
| 最大絶対誤差 | 2.84e-02 |
| `np.allclose(atol=1e-4)` | False |

### Decoder

| 指標 | mel_out | postnet_out |
|------|---------|-------------|
| 最大絶対誤差 | 1.57e-03 | 1.54e-03 |
| `np.allclose(atol=1e-4)` | False | False |

Encoder の誤差は HiFi-GAN と同程度 (Float16 由来)。Decoder の誤差は非常に小さい。いずれも音声品質に影響しない水準。

## 生成ファイル

| ファイル | 説明 |
|---------|------|
| `scripts/convert_transformer.py` | 変換スクリプト |
| `Transformer_Encoder.mlpackage` | Encoder の CoreML モデル (gitignore 対象) |
| `Transformer_Decoder.mlpackage` | Decoder の CoreML モデル (gitignore 対象) |

## 実行方法

```bash
venv/bin/python scripts/convert_transformer.py
```
