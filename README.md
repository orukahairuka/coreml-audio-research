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

## 元リポジトリ

- **PronounSE**: https://github.com/Jinmaro/PronounSE
  - セットアップ・学習・推論の詳細はこちらを参照
