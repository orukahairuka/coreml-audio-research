# coreml-audio-research

[PronounSE](https://github.com/Jinmaro/PronounSE)（声真似→効果音合成）の学習済みモデルを [coremltools](https://github.com/apple/coremltools) で CoreML 形式に変換し、iOS 上でオンデバイス推論できるようにするリポジトリ。

## 構成

```
scripts/              # PyTorch → CoreML 変換スクリプト
  convert_transformer.py
  convert_hifigan.py
  synthesis_coreml.py  # CoreML モデルでの推論確認用
ios/CoreMLAudioApp/   # iOS アプリ (Swift / SwiftUI)
PronounSE/            # 元リポジトリ (git submodule)
```

## 変換対象モデル

| モデル | 説明 |
|---|---|
| Transformer Encoder | 入力音声→メルスペクトログラムの特徴量抽出 |
| Transformer Decoder | 特徴量→変換メルスペクトログラム |
| HiFi-GAN Generator | メルスペクトログラム→波形（ボコーダー） |

## モデルファイル

変換済み CoreML モデル（`*.mlpackage`）はリポジトリには含まれていません。Releases から `mlpackages.zip` をダウンロードして展開してください。

```bash
# Releases から取得して展開
unzip mlpackages.zip -d ios/CoreMLAudioApp/CoreMLAudioApp/MLModels/
```

含まれるバリアント（全24個）：

| モデル | 精度 | 入力長バリアント |
|---|---|---|
| Transformer Encoder / Decoder | float32 / float16 / int8 | range1 / fixed262 |
| HiFi-GAN Generator | float32 / float16 / int8 | range1 / range16 / range16_384 / fixed262 |

本番用途は `*_fixed262` 系（合計 ~320MB）。`range1`, `range16`, `range16_384` は量子化と入力長の比較実験用です。

## 元リポジトリ

- **PronounSE**: https://github.com/Jinmaro/PronounSE
  - セットアップ・学習・推論の詳細はこちらを参照
  - 学習済みチェックポイント（`chkpt__20000.pth.tar`, `g_00009000`）も Jinmaro 側 Releases から取得

## ライセンス

MIT License。本リポジトリは [PronounSE](https://github.com/Jinmaro/PronounSE)（MIT License, Copyright (c) 2025 Riki Takizawa）をベースにしています。
