# coreml-audio-research

[PronounSE](https://github.com/Jinmaro/PronounSE)（声真似→効果音合成）の学習済みモデルを [coremltools](https://github.com/apple/coremltools) で CoreML 形式に変換し、iPhone 上でオンデバイス推論する研究リポジトリ。

PronounSE を iPhone 上に実装し、実行構成（数値形式 FP32 / FP16 / INT8 と計算ユニットの指定）と入力音声の長さによる
**推論性能**の違いと、実行構成による**合成品質**（FAD）の違いを実機で明らかにする。

## 構成

```
scripts/              # 変換・実機計測・集計・検証のスクリプト（索引は scripts/README.md）
eval/fad/             # FAD（音質）の評価環境。データセットと正式評価の設定は eval/fad/README.md
metrics/              # 実機の計測結果（正式評価は formal_eval_official/・fad_official_fadtk/・rtf_official/）
docs/                 # 日付ごとの作業メモ・レポート（目次は docs/README.md）
ios/CoreMLAudioApp/   # iOS アプリ（Swift / SwiftUI）
PronounSE/            # 元リポジトリ（git submodule）
```

## 変換対象モデル

| モデル | 説明 |
|---|---|
| Transformer Encoder | 入力音声のメルスペクトログラム→特徴量 |
| Transformer Decoder | 特徴量→変換後のメルスペクトログラム（自己回帰で T 回まわす） |
| HiFi-GAN Generator | メルスペクトログラム→波形（ボコーダー） |

変換は `scripts/convert_transformer.py`・`scripts/convert_hifigan.py`。
CoreML の後継の Core AI（`.aimodel`）への変換も `scripts/convert_*_coreai.py` で試している。

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

用途は次のとおり。

| バリアント | 用途 |
|---|---|
| `range1`（RangeDim 1〜1000 フレーム ≒ 11.6 秒） | **正式評価（FAD・RTF）**。入力を切らずに全体を合成する |
| `fixed262`（262 フレーム ≒ 3 秒に固定） | 卒業論文第 4 章の計測（`ThesisBenchConfig`）と、アプリの通常の録音→合成 |
| `range16`, `range16_384` | 入力長の比較実験用 |

## 正式評価（2026-11）

オリジナル PronounSE の開発者から提供された評価データセット（再配布しないので Git には入れない）を使い、
オリジナルと同じ入力・前処理・後処理で iPhone 17 Pro 上で合成して比べる。

| 構成 | 精度 | Encoder | Decoder | HiFi-GAN |
|---|---|---|---|---|
| A | FP32 | cpuAndGPU | cpuAndGPU（padT） | cpuOnly |
| B | FP16 | cpuAndGPU | cpuOnly | cpuAndGPU |
| C | INT8（W8A16） | cpuOnly | cpuOnly | cpuOnly |

条件は `range1` ＋ 前処理 `fullInput`（先頭の無音を削らない・フェードしない）＋ 22050 Hz の float32 wav 入力。
48 kHz の wav をそのまま iOS に渡すと、標本化周波数の変換の違いで別の音になる。詳細と根拠は `eval/fad/README.md`。

### 流れ

1. **入力を作る**: `scripts/prepare_formal_eval_inputs.py` で口真似音を 22050 Hz の float32 wav にする（`eval/fad/data/formal_inputs/`）
2. **実機で一括合成（FAD 用）**: `scripts/run_formal_eval.py` が入力を端末に送り、アプリを `CMLA_FORMAL_EVAL=1`（`FormalEvalRunner`）で起動して結果を取り出す
3. **検証**: `scripts/verify_formal_session.py`（件数・長さ・NaN などを全件確認）、`scripts/compare_formal_to_pytorch.py`（PyTorch FP32 との比較）
4. **FAD**: `eval/fad/run_formal_fadtk.py`（DCASE2024 Task7 版 fadtk、PANNs Wavegram-Logmel）。結果は `metrics/fad_official_fadtk/`
5. **RTF**: `scripts/rtf_official/run_rtf_official.py` がアプリを `CMLA_RTF_BENCH=1`（`RTFBenchRunner`、Release ビルド）で起動する。結果は `metrics/rtf_official/`。集計は `scripts/rtf_official/aggregate_rtf.py`、図4-1 は `scripts/rtf_official/make_thesis_figure.py`

各スクリプトの使い方は、ファイル先頭の docstring に書いてある。結果の数値は `metrics/` 以下の各 README を参照。

## 元リポジトリ

- **PronounSE**: https://github.com/Jinmaro/PronounSE
  - セットアップ・学習・推論の詳細はこちらを参照
  - 学習済みチェックポイント（`chkpt__20000.pth.tar`, `g_00009000`）も Jinmaro 側 Releases から取得

## ライセンス

MIT License。本リポジトリは [PronounSE](https://github.com/Jinmaro/PronounSE)（MIT License, Copyright (c) 2025 Riki Takizawa）をベースにしています。
