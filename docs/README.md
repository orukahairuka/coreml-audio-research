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

### 理論・前提知識メモ

- [fft-vs-stft](2026-04-05/fft-vs-stft.md)
