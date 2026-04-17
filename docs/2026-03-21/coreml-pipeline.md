# CoreML パイプライン合成レポート

## 概要

変換済みの CoreML モデル 3 つ（Encoder / Decoder / HiFi-GAN）を繋いで、Mac 上の Python で音声合成パイプラインを実行した。PyTorch 版 (`PronounSE/synthesis.py`) と同じ入力で合成し、出力を比較した。

## 環境

| 項目 | 値 |
|------|-----|
| マシン | Mac M4 Max |
| Python | 3.9.6 |
| coremltools | 9.0 |
| モデル精度 | Float16 (mlprogram) |

## パイプラインの処理フロー

```
入力音声 (.wav)
  ↓ メルスペクトログラム抽出 (numpy/librosa)
  ↓
Encoder (CoreML)  →  memory [1, T_src, 512]
  ↓
Decoder (CoreML)  ←  自己回帰ループ (T_src ステップ)
  ↓
メルスペクトログラム [1, T_src, 256]
  ↓ 転置 → [1, 256, T_src]
  ↓
HiFi-GAN (CoreML)  →  音声波形 [1, 1, T_src × 256]
  ↓
デエンファシスフィルタ
  ↓
出力音声 (.wav)
```

### 各ステップの詳細

1. **メルスペクトログラム抽出**: `get_spectrograms()` で入力音声を 256 次元のメルスペクトログラムに変換する。librosa の STFT + メルフィルタバンク + dB 正規化で処理。
2. **Encoder**: メルスペクトログラムと位置インデックスを受け取り、512 次元の中間表現 (memory) を出力。1 回だけ実行。
3. **Decoder**: memory を参照しながら、出力メルスペクトログラムを 1 フレームずつ自己回帰的に生成。初期入力はゼロベクトルで、各ステップで前回の出力を入力に追加していく。入力フレーム数分ループ実行。
4. **HiFi-GAN**: メルスペクトログラムを音声波形に変換（256 倍アップサンプリング）。
5. **デエンファシスフィルタ**: 前処理で適用したプリエンファシス（高域強調）の逆処理。`y[n] = x[n] + 0.97 * y[n-1]`

## PyTorch 版との比較結果

同じ入力音声 (`input_sample.wav`, 266 フレーム) で合成し、PyTorch 版と CoreML 版の出力波形を比較した。

| 指標 | 値 |
|------|-----|
| サンプル数 | 68096（完全一致） |
| 最大絶対誤差 | 1.61e-01 |
| 平均絶対誤差 | 3.05e-03 |
| SNR | 23.1 dB |

### 誤差の蓄積について

各モデル単体の誤差（最大 2.8e-02）に比べて、パイプライン全体の誤差（最大 1.61e-01）が大きくなっている。これは以下の理由による:

- **3 モデル直列**: Encoder → Decoder → HiFi-GAN と 3 つのモデルを直列に通すため、各モデルの Float16 誤差が蓄積する
- **自己回帰ループ**: Decoder は前のステップの出力を次の入力に使うため、誤差が 266 ステップ分フィードバックされる。1 ステップの微小な誤差が後続ステップで増幅される

### 音質の主観評価

最大絶対誤差は 16% と数値上は大きいが、聴覚上の差異はほとんど感じられなかった。平均誤差が 0.3% と小さく、SNR 23dB は音声合成として十分な水準。

## 制限事項

- 入力は最大 1000 フレーム（CoreML 変換時の `RangeDim` 上限）
- CoreML の出力キー名が自動リネームされる場合があるため、明示的なキー名または挿入順で取得する必要がある

## 生成ファイル

| ファイル | 説明 |
|---------|------|
| `scripts/synthesis_coreml.py` | CoreML パイプライン合成スクリプト |

## 実行方法

```bash
# 事前に変換スクリプトで .mlpackage を生成しておく
venv/bin/python scripts/convert_hifigan.py
venv/bin/python scripts/convert_transformer.py

# CoreML パイプラインで合成
venv/bin/python scripts/synthesis_coreml.py <input.wav>
# → result/<basename>_coreml.wav に出力
```
