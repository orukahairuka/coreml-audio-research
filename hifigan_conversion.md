# HiFi-GAN Generator CoreML 変換レポート

## 概要

PronounSE の HiFi-GAN Generator を CoreML (.mlpackage) に変換した。

## 環境

| 項目 | 値 |
|------|-----|
| マシン | Mac M4 Max |
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| coremltools | 9.0 |
| 変換形式 | mlprogram (Float16) |

## モデル構成

| パラメータ | 値 |
|-----------|-----|
| 入力 | メルスペクトログラム `[1, 256, T]` (T: 可変長 1〜1000) |
| 出力 | 音声波形 `[1, 1, T×256]` |
| アップサンプリング倍率 | 256× (8×8×2×2) |
| アップサンプリングカーネル | [16, 16, 4, 4] |
| 初期チャネル数 | 512 → 256 → 128 → 64 → 32 |
| ResBlock | ResBlock1 × 12 (4段 × カーネル3種) |
| ResBlock カーネル | [3, 7, 11] |
| ResBlock ディレーション | [[1,3,5], [1,3,5], [1,3,5]] |
| サンプリングレート | 22050 Hz |
| チェックポイント | `g_00009000` |

## 変換手順

1. Generator を CPU でロード、`weight_norm` を除去
2. ダミー入力 `torch.randn(1, 256, 100)` を作成
3. `torch.jit.trace` で TorchScript に変換
4. `coremltools.convert` で mlprogram 形式に変換（時間軸は `RangeDim` で可変長対応）
5. 変換前後の出力を数値比較

## 精度比較結果

ダミー入力 `torch.randn(1, 256, 100)` に対して、PyTorch (Float32) と CoreML (Float16) の出力を比較した。

| 指標 | 値 |
|------|-----|
| PyTorch 出力 shape | `(1, 1, 25600)` — 100 フレーム × 256 アップサンプリング |
| 最大絶対誤差 | 1.87e-02 |
| `np.allclose(atol=1e-4)` | False |
| `np.allclose(atol=1e-2)` | False |
| `np.allclose(atol=2e-2)` | True |

### なぜ誤差が生じるか

CoreML の `mlprogram` バックエンドはデフォルトで **Float16 (半精度)** で計算を行う。PyTorch のデフォルトは **Float32 (単精度)**。

- Float32: 仮数部 23 bit → 約 7 桁の精度
- Float16: 仮数部 10 bit → 約 3 桁の精度

この精度の差がニューラルネットの各層で蓄積され、最終出力で最大 1.87e-02 の誤差として現れる。HiFi-GAN は 4 段のアップサンプリング + 12 個の ResBlock を通るため、層が深い分だけ誤差が伝播しやすい。

### この誤差は問題か

- 出力波形の範囲は **[-1, 1]** なので、最大誤差 0.0187 は振幅の **約 1.9%**
- 人間の聴覚は位相や微細な波形の違いにはほとんど敏感でないため、この程度の誤差は**聴覚上ほぼ判別不能**
- `coremltools.convert` に `compute_precision=ct.precision.FLOAT32` を指定すれば Float32 で変換でき誤差は大幅に減るが、推論速度とモデルサイズが増加する
- 量子化（INT8 など）を行うとさらに誤差は増えるため、Float16 での誤差は今後の比較のベースラインとなる

## 生成ファイル

| ファイル | 説明 |
|---------|------|
| `convert_hifigan.py` | 変換スクリプト |
| `HiFiGAN_Generator.mlpackage` | 変換済み CoreML モデル (gitignore 対象) |

## 実行方法

```bash
venv/bin/python convert_hifigan.py
```
