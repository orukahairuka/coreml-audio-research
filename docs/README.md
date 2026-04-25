# docs/

CoreML 変換・iOS 実装の研究進捗ドキュメント。日付ディレクトリ配下に作業日ごとのメモを置く。

配置規約は [`.claude/rules/docs-structure.md`](../.claude/rules/docs-structure.md) を参照。

## 時系列（進捗）

| 日付 | ドキュメント | 内容 |
|------|------|------|
| 2026-03-21 | [coreml-pipeline](2026-03-21/coreml-pipeline.md) | CoreML 3 モデルを Python で繋いだ合成パイプライン検証 |
| 2026-03-21 | [hifigan-conversion](2026-03-21/hifigan-conversion.md) | HiFi-GAN Generator の CoreML 変換レポート |
| 2026-03-21 | [transformer-conversion](2026-03-21/transformer-conversion.md) | Transformer Encoder/Decoder の CoreML 変換レポート |
| 2026-04-05 | [fft-vs-stft](2026-04-05/fft-vs-stft.md) | FFT と STFT の違いを整理した理論メモ |
| 2026-04-05 | [float32-gpu-debug-report](2026-04-05/float32-gpu-debug-report.md) | Float32 + GPU/ANE 実行時の音声異常 調査レポート |
| 2026-04-05 | [ios-app-implementation](2026-04-05/ios-app-implementation.md) | iOS アプリ (CoreMLAudioApp) の実装詳細 |
| 2026-04-17 | [coreml-api-notes](2026-04-17/coreml-api-notes.md) | CoreML API の典型パターンを EncoderRunner 題材に整理 |
| 2026-04-17 | [models-refactor](2026-04-17/models-refactor.md) | iOS 合成パイプラインのリファクタ作業セッションメモ |
| 2026-04-22 | [decoder-runner-breakdown](2026-04-22/decoder-runner-breakdown.md) | DecoderRunner.swift を【T】【C】【U】【D】の 4 種類に色分けして読み解くメモ |
| 2026-04-22 | [coreml-wrapping-analysis](2026-04-22/coreml-wrapping-analysis.md) | CoreML のラップ・型変換を全種類洗い出し、1 秒合成あたり約 114 万回発生していることを集計 |
| 2026-04-22 | [research-plan](2026-04-22/research-plan.md) | ここまでの前提スナップショット + 先生相談アジェンダ（論点・仮案・音質指標カタログ） |
| 2026-04-22 | [advisor-meeting-todo](2026-04-22/advisor-meeting-todo.md) | 先生相談後の持ち帰り TODO（Float32/GPU、HiFi-GAN 差し替え、比較指標、データ・文献） |
| 2026-04-25 | [timing-measurement](2026-04-25/timing-measurement.md) | CoreML 各段の `predict()` 所要時間を計測する機構の実装メモ |
| 2026-04-25 | [ui-test-loop-fixes](2026-04-25/ui-test-loop-fixes.md) | 12通りバッチ取得 XCUITest のフレーク対策3点（XCTFail 二重計上 / picker disabled 残留 / status ラベル即マッチ）の解説 |

## トピック別

### CoreML 変換

- [transformer-conversion](2026-03-21/transformer-conversion.md)
- [hifigan-conversion](2026-03-21/hifigan-conversion.md)
- [coreml-pipeline](2026-03-21/coreml-pipeline.md)
- [float32-gpu-debug-report](2026-04-05/float32-gpu-debug-report.md)

### iOS 実装

- [ios-app-implementation](2026-04-05/ios-app-implementation.md)
- [coreml-api-notes](2026-04-17/coreml-api-notes.md)
- [models-refactor](2026-04-17/models-refactor.md)
- [decoder-runner-breakdown](2026-04-22/decoder-runner-breakdown.md)
- [coreml-wrapping-analysis](2026-04-22/coreml-wrapping-analysis.md)
- [timing-measurement](2026-04-25/timing-measurement.md)
- [ui-test-loop-fixes](2026-04-25/ui-test-loop-fixes.md)

### 理論・前提知識メモ

- [fft-vs-stft](2026-04-05/fft-vs-stft.md)
- [decoder-runner-breakdown](2026-04-22/decoder-runner-breakdown.md)

### 研究方針

- [research-plan](2026-04-22/research-plan.md)
- [advisor-meeting-todo](2026-04-22/advisor-meeting-todo.md)
