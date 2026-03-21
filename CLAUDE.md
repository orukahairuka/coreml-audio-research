# coreml-audio-research

CoreML 変換・量子化と音質トレードオフの研究リポジトリ。PronounSE（声真似→効果音合成）をサブモジュールとして使用。

## 環境

- Python 3.9.6 + venv (`PronounSE/venv/`)
- PyTorch 2.8.0 (MPS対応 Mac M4 Max)、coremltools 9.0
- CUDA なし。`.cuda()` は `.to(DEVICE)` に修正済み

## 実行

```bash
cd PronounSE && venv/bin/python synthesis.py <input.wav>
# 結果は PronounSE/result/ に出力される
```

## PronounSE ファイル構成

- `PronounSE/synthesis.py` — 合成パイプライン（エントリーポイント）
- `PronounSE/Transformer/network.py` — Encoder/Decoder モデル
- `PronounSE/Transformer/module.py` — Attention, FFN, Prenet 等
- `PronounSE/Transformer/hyperparams.py` — ハイパーパラメータ (n_mels=256, hidden_size=512, sr=22050)
- `PronounSE/Transformer/utils.py` — 前処理、メルスペクトログラム変換
- `PronounSE/HiFiGAN/models.py` — Generator（ボコーダー）
- `PronounSE/HiFiGAN/chkpt/config.json` — HiFi-GAN 設定

## チェックポイント

- `PronounSE/Transformer/chkpt/chkpt__20000.pth.tar`
- `PronounSE/HiFiGAN/chkpt/g_00009000`
- .gitignore 対象。GitHub Releases から取得する

## 修正履歴（PronounSE）

- `synthesis.py`: `.cuda()` → `.to(DEVICE)`, `map_location=DEVICE` 追加, plot関数のゼロパディングバグ修正
- `Transformer/network.py`: MelDecoder.forward 内の `.cuda()` → `device=memory.device`

## 注意

- PronounSE は git submodule（元は Jinmaro のリポジトリ）
- チェックポイント (`*.pth`, `*.tar`, `g_*`), `result/`, `venv/`, `my_voice.wav` は gitignore 対象

## Git運用ルール

### ブランチ
- main に直接コミット・push しない。必ずブランチを切って PR 経由でマージする
- ブランチ名は `feature/` に統一する（`fix/` `docs/` などは使わない）
- 同じスコープの変更は同じブランチで行う（レビュー修正やレポート追加も同じブランチ）

### コミット
- コミットは小さめの粒度で行う
- コミットを実行する前に、必ず実行してよいか y/n で確認する
- コミットメッセージは日本語で書く
- `feat:` `fix:` などのプレフィックスは付けない
- 関連のない変更を1つのコミットにまとめない

### PR
- PR を立てる前に必ず動作確認（スクリプト実行・ビルド等）を行う
