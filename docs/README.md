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
| 2026-04-27 | [quantization-pareto-baseline](2026-04-27/quantization-pareto-baseline.md) | 量子化 × 計算デバイス 12 通りのサイズ・速さ・mel 劣化を1表にしたシミュレータ計測ベースライン |
| 2026-04-27 | [float32-gpu-accumulation-experiment](2026-04-27/float32-gpu-accumulation-experiment.md) | Float32 × GPU 出力飽和の原因切り分け：累積精度 fp32 強制 (`allowLowPrecisionAccumulationOnGPU = false`) では直らず仮説 1 否定、中間テンソル fp16 保持仮説も Apple 公式 Typed Execution と整合せず撤回。**原因未特定**で追加調査の方針を残した |
| 2026-05-06 | [float32-gpu-investigation-summary](2026-05-06/float32-gpu-investigation-summary.md) | `feature/float32-gpu-investigation` ブランチ ([PR #15](https://github.com/orukahairuka/coreml-audio-research/pull/15)) の本文と 21 コミットの全体まとめ（計測機構 / 12 通りベースライン / Float32×GPU 飽和調査 / fixed262 採用） |
| 2026-05-07 | [hirai-comment-memory-management](2026-05-07/hirai-comment-memory-management.md) | RangeDim+GPU の E5RT 問題に関する平井先生 Slack コメントの解釈メモ（動的メモリ確保の観点、先生の 3 つの意図整理、section 3/4 との接続） |
| 2026-05-08 | [device-benchmark-workflow](2026-05-08/device-benchmark-workflow.md) | iPhone 実機で 12 通り計測を 1 コマンドで完走させるワークフロー（`run_device_benchmark.sh`、`xcrun devicectl` 経由のファイル吸い出し） |
| 2026-05-10 | [research-direction](2026-05-10/research-direction.md) | `/grill-me` セッションを通じた卒研方針の整理。研究目標・公知化する 5 貢献・章立て・採用したアプリスコープ・採用しなかった案の理由・次に決めること |
| 2026-05-17 | [fp32-cpuandgpu-quiet-vs-loud-investigation](2026-05-17/fp32-cpuandgpu-quiet-vs-loud-investigation.md) | XCUITest 経由だと F32 × cpuAndGPU が quiet（rms 400-728）、手動操作だと loud（rms 5029、決定論的）の調査メモ。playback-wait race 修正、順序効果（cpuAndGPU を先頭にすると後続 cpuOnly/cpuAndNE/all が loud に化ける）、未確定の論点を中間まとめ |
| 2026-05-19 | [all-engine-precision-stability-plan](2026-05-19/all-engine-precision-stability-plan.md) | 全 12 組合せ（precision × computeUnits）で爆発音を安定して鳴らす条件を体系的に探索する Phase 0〜5 の研究計画。マトリクス計測 → 救済実験 → MLComputePlan → Decoder 内部分解 → 実装方針 |
| 2026-05-19 | [stability-matrix-results](2026-05-19/stability-matrix-results.md) | Phase 1 の生データ集計（aggregate_stability_matrix.py 自動生成） |
| 2026-05-19 | [stability-matrix-analysis](2026-05-19/stability-matrix-analysis.md) | Phase 1 実機の観測整理。F16/Int8×{cpuAndNE,all} で clipping を観測、F32×cpuAndGPU 単発初回の非決定性を再現、warm-up の効果を確認 |
| 2026-05-19 | [mlcomputeplan-dispatch-map](2026-05-19/mlcomputeplan-dispatch-map.md) | Phase 4 の生データ集計（18 plans、aggregate_compute_plan.py 自動生成） |
| 2026-05-19 | [compute-plan-analysis](2026-05-19/compute-plan-analysis.md) | Phase 4 の dispatch 観測整理。F16/Int8×{cpuAndNE,all} で Decoder/HiFi-GAN 両方に NE dispatch、HiFi-GAN 出力時点で振幅差が観測される。op 単位の原因は未確定 |
| 2026-05-19 | [coreml-hifigan-investigation-handoff](2026-05-19/coreml-hifigan-investigation-handoff.md) | /clear 後に調査を再開するための引き継ぎノート。確定事実・未確定事項・次にやることを整理 |
| 2026-05-19 | [phase2-mini-rescue-results](2026-05-19/phase2-mini-rescue-results.md) | Phase 2 mini 救済実験。出力正規化（J）と HiFi-GAN だけ cpuAndGPU 退避（K）の効果を実機で計測。K は Decoder=ANE postnet sha が cpuAndNE 単独と bit-identical、HiFi-GAN dispatch だけで clipping が消える観測 |
| 2026-05-19 | [phase5-implementation-policy-draft](2026-05-19/phase5-implementation-policy-draft.md) | Phase 5 実装方針のドラフト。Phase 1/2 mini/4 の観測を踏まえた設定別方針案・UI 設計案・残課題（聴感判定）の整理。確定ではない |
| 2026-06-10 | [load-timing-results](2026-06-10/load-timing-results.md) | 12セルの3モデルロード時間を「初回/キャッシュ後」で実機計測（n=2）。ANE系HiFi-GAN(F16/Int8×{cpuAndNE,all})の初回特殊化が32-67秒or時々クラッシュ、CPU/GPU系8セルは初回約1秒、キャッシュ後は全セル約0.1秒。clippingで壊れる4セルと一致し「速い」軸と「壊れない」軸が同じANEセルを同時棄却 |

## トピック別

### CoreML 変換

- [transformer-conversion](2026-03-21/transformer-conversion.md)
- [hifigan-conversion](2026-03-21/hifigan-conversion.md)
- [coreml-pipeline](2026-03-21/coreml-pipeline.md)
- [float32-gpu-debug-report](2026-04-05/float32-gpu-debug-report.md)
- [float32-gpu-accumulation-experiment](2026-04-27/float32-gpu-accumulation-experiment.md)
- [fp32-cpuandgpu-quiet-vs-loud-investigation](2026-05-17/fp32-cpuandgpu-quiet-vs-loud-investigation.md)
- [all-engine-precision-stability-plan](2026-05-19/all-engine-precision-stability-plan.md)

### iOS 実装

- [ios-app-implementation](2026-04-05/ios-app-implementation.md)
- [coreml-api-notes](2026-04-17/coreml-api-notes.md)
- [models-refactor](2026-04-17/models-refactor.md)
- [decoder-runner-breakdown](2026-04-22/decoder-runner-breakdown.md)
- [coreml-wrapping-analysis](2026-04-22/coreml-wrapping-analysis.md)
- [timing-measurement](2026-04-25/timing-measurement.md)
- [ui-test-loop-fixes](2026-04-25/ui-test-loop-fixes.md)
- [device-benchmark-workflow](2026-05-08/device-benchmark-workflow.md)

### 理論・前提知識メモ

- [fft-vs-stft](2026-04-05/fft-vs-stft.md)
- [decoder-runner-breakdown](2026-04-22/decoder-runner-breakdown.md)

### 研究方針

- [research-plan](2026-04-22/research-plan.md)
- [advisor-meeting-todo](2026-04-22/advisor-meeting-todo.md)
- [research-direction](2026-05-10/research-direction.md)
- [hirai-comment-memory-management](2026-05-07/hirai-comment-memory-management.md)
- [all-engine-precision-stability-plan](2026-05-19/all-engine-precision-stability-plan.md)
- [phase5-implementation-policy-draft](2026-05-19/phase5-implementation-policy-draft.md)

### 計測・評価結果

- [quantization-pareto-baseline](2026-04-27/quantization-pareto-baseline.md)
- [stability-matrix-analysis](2026-05-19/stability-matrix-analysis.md)
- [compute-plan-analysis](2026-05-19/compute-plan-analysis.md)
- [phase2-mini-rescue-results](2026-05-19/phase2-mini-rescue-results.md)
- [load-timing-results](2026-06-10/load-timing-results.md)

### ブランチ・PR まとめ

- [float32-gpu-investigation-summary](2026-05-06/float32-gpu-investigation-summary.md)
