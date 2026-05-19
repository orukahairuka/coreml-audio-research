# 全 engine × precision 合成安定性 調査計画（2026-05-19）

ブランチ: `feature/fp32-gpu-memory-timing`

## 0. 動機と目的

これまでの調査（[2026-05-17 まとめ](../2026-05-17/fp32-cpuandgpu-quiet-vs-loud-investigation.md)、[F32×cpuAndGPU 2nd-call 収束メモ（記憶）]）で次が判明している:

- F32 × cpuAndGPU の 1st call は **非決定的**（quiet / clipped / loud）
- 同インスタンスを使い回すと 2nd call 以降 manual と bit-identical な loud（rms 5029 / peak 24326）に収束
- 異常 run の Decoder step 1 で `mel_min=inf, mel_max=-inf, mel_mean=nan` を観測 → Decoder 内部で NaN/Inf 発火
- F16 × cpuAndGPU, Int8 × cpuAndGPU, F32 × cpuOnly, F32 × all は iter1 から決定論的に loud
- mel_normalized / encoder_output は全 run で一致 → divergence は Decoder loop で起きている
- XCUITest / AVAudioSession / HiFi-GAN 単体は主原因ではない

ここで「F32 × cpuAndGPU を本番で避ける」という実用結論は出せるが、研究としてはそこで打ち切らない。

### 本調査の研究目的

1. **全 12 組合せ（3 precision × 4 computeUnits）で「爆発音が正常に鳴る条件」を体系的に同定する**
2. F32 × cpuAndGPU の Decoder 初回 NaN/Inf 発火点を特定する（どの layer / op か）
3. 救済策（warm-up / retry / fallback / shape）の効果率を計測する
4. 上記をもとに「アプリで爆発音を必ず鳴らす実装方針」を確定する

### 非目標

- Apple の閉じた GPU runtime そのものの修正
- 全 op の完全な数値解析（Phase 3 では「最初の divergence 発火層」までを特定する深さに留める）

## 1. Phase 構成と着手順序

| Phase | 内容 | 成果物 | 着手順 |
|---|---|---|---|
| 0 | 本計画書 | `docs/2026-05-19/all-engine-precision-stability-plan.md` | 1 |
| 1 | 全 12 組合せ安定性マトリクス計測 | `docs/2026-05-19/stability-matrix-results.md` + CSV | 2 |
| 2 | 救済実験 A〜I マトリクス | `docs/2026-05-19/rescue-experiment-results.md` | 3 |
| 4 | MLComputePlan で device assignment | `docs/2026-05-19/mlcomputeplan-dispatch-map.md` | 4 |
| 3 | Decoder 内部 divergence point 特定 | `docs/2026-05-19/decoder-internal-divergence.md` + npy | 5 |
| 5 | 実装方針の確定 | `docs/2026-05-19/synthesis-stability-implementation-plan.md` | 6 |

Phase 4 を Phase 3 より先に置くのは、device assignment を先に取れば Phase 3 で「どの op の中間 tensor を見ればよいか」の当たりが付くため。

## 2. Phase 1 — 全 12 組合せ安定性マトリクス

### 2.1 軸

- precision: `Float32` / `Float16` / `Int8` (`ModelPrecision`)
- computeUnits: `cpuOnly` / `cpuAndGPU` / `cpuAndNE` / `all` (`ComputeUnitOption`)
- = **12 組合せ**

### 2.2 計測単位

各組合せで:

- **連続呼出し系列**: 同一 `AudioSynthesizer` インスタンスで iter1 → iter2 → iter3
- **fresh launch 系列**: 別 launch を 3 回（既存 XCUITest を流用）

→ 1 組合せあたり 6 run、12 組合せで **72 run**。これを 1 ラウンドとする。

将来的に N=10 fresh launch も取りたいが、まずは 1 ラウンドで全体像を取る。

### 2.3 各 run で記録する項目

| 項目 | 取得元 |
|---|---|
| `int16_rms` | postdeemph 波形（既存実装） |
| `int16_peak` | postdeemph 波形 |
| `wav_sha256` | postdeemph 波形 |
| `mel_normalized_sha256` | `DebugRunSnapshot` |
| `encoder_output_sha256` | `DebugRunSnapshot` |
| `postnet_output_sha256` | `DebugRunSnapshot` |
| `waveform_postdeemph_sha256` | `DebugRunSnapshot` |
| `decoder_step1_min/max/mean` | `decoder_steps.csv` |
| `has_nan` / `has_inf` | postnet_output と waveform |
| `classification` | 後述ルールで自動付与 |
| `predict_total_ms` | Encoder + Decoder + Vocoder の合計 |
| `precision` / `computeUnits` / `iter` / `launch_id` | メタ |

### 2.4 分類ルール

run ごとに次のいずれかを付与する（先に当てはまるものを優先）:

1. `predict_failed`: 例外で落ちた
2. `nan_inf`: `has_nan == true` or `has_inf == true`
3. `clipped`: `int16_peak > 32000` かつ `int16_rms > 7000`
4. `quiet`: `int16_rms < 3000`
5. `normal_loud`: 上記以外（参考: manual baseline `rms ≈ 5029, peak ≈ 24326`）

### 2.5 実装

- 既存 `Fp32QuietInvestigationTests.swift` を拡張し、12 組合せ全ての Repeat3 ケースを追加（F32 の 4 通り、F16 の 4 通り、Int8 の 4 通り）
- 既存 XCUITest `testCaptureAllCombinations` 系を流用して fresh launch シーケンスを取得
- 結果集約スクリプト `scripts/aggregate_stability_matrix.py` を追加し、`Documents/Result/debug/*/summary.json` + `decoder_steps.csv` から CSV を生成

### 2.6 成果物

- `docs/2026-05-19/stability-matrix-results.md`
  - 12 組合せ × {iter1, iter2, iter3, launch1, launch2, launch3} のヒートマップ表
  - 分類別カウント
  - 観察事項
- `data/2026-05-19/stability_matrix.csv`（生データ）

## 3. Phase 2 — 救済実験マトリクス

Phase 1 で `normal_loud` 以外が出た組合せに対し、以下の戦略の効果率を測る。
A〜I は F32 × cpuAndGPU の 1st call 非決定性を主眼にした warm-up / retry / fallback 系。
J / K は F16/Int8 × cpuAndNE の clipping を救えるかを見る Phase 2 mini で 2026-05-19 に追加した戦略。

| ID | 戦略 | 実装の要点 | 主対象 |
|---|---|---|---|
| A | dummy warm-up 1 回 → 本番 | `loadModels` 直後に zero 入力で 1 回 synthesize。本番は wav を採用 | F32 × cpuAndGPU |
| B | dummy warm-up 2 回 → 本番 | A の dummy を 2 連続 | F32 × cpuAndGPU |
| C | warm-up → 検査 → retry | dummy 後 / 本番後の rms/NaN/Inf を見て異常なら再合成（最大 N 回） | F32 × cpuAndGPU |
| D | Decoder のみ短い dummy で warm-up | Encoder/HiFi-GAN は skip し Decoder model だけ短メル長で 1 回 predict | F32 × cpuAndGPU |
| E | Encoder / Decoder / HiFi-GAN を個別 warm-up | 3 段独立に warm-up し、どの段の warm-up が効くかを切り分け | F32 × cpuAndGPU |
| F | cpuOnly で warm-up → cpuAndGPU で本番 | `MLModelConfiguration` を 2 つ作り、warm-up 用と本番用で切替 | F32 × cpuAndGPU |
| G | インスタンス再生成 vs 使い回し | 各 iter で `AudioSynthesizer` を再 init するパターンを別途取る | F32 × cpuAndGPU |
| H | fixed262 vs RangeDim 比較 | `ShapeModeOption` を切り替えた Repeat3 を取る | F32 × cpuAndGPU |
| I | 別 precision/engine を先に走らせた後に対象を走らせる | プロセス全体ウォーム状態の影響を測る | F32 × cpuAndGPU |
| J | 出力波形を後処理で再正規化 | `outputWaveform` を peakNormalize(0.95) / rmsNormalize(baseline 5029/32767) / fixedGain ×0.25 の 3 通りで test 側で再正規化し、それぞれを別 runId の snapshot として書く | F16/Int8 × cpuAndNE |
| K | HiFi-GAN だけ別エンジンで再ロード | `AudioSynthesizer.reloadHifigan(precision, computeUnits, shapeMode)` で Encoder/Decoder は loadModels 時の computeUnits のまま、HiFi-GAN だけ別 engine に載せる | F16/Int8 × cpuAndNE → HiFi-GAN を cpuAndGPU に逃がす |

A〜I は計画段階の N=20 を想定。J / K は実機 1 ラウンド計測（J=4 synth、K=6 synth）で観測した。結果は [`phase2-mini-rescue-results.md`](phase2-mini-rescue-results.md) を参照。

### 3.1 成果物

- `docs/2026-05-19/rescue-experiment-results.md`
  - 戦略 × 対象組合せ の効果率マトリクス
  - 各戦略の副作用（起動コスト、メモリ、コード複雑度）の所見

## 4. Phase 4 — MLComputePlan で device assignment 取得

iOS 17+ の [`MLComputePlan`](https://developer.apple.com/documentation/coreml/mlcomputeplan) で、Decoder model の各 op が CPU / GPU / NE のどこに dispatch されるかを列挙する。

### 4.1 取得項目

- 各 op の `MLComputeDeviceUsage`
- `estimatedCost` / `estimatedTime`
- op 種別（matmul / layernorm / softmax / activation など）

### 4.2 比較対象

- F32 × cpuAndGPU の Decoder dispatch map
- F32 × cpuOnly の Decoder dispatch map
- F16 × cpuAndGPU の Decoder dispatch map

特に F32 × cpuAndGPU で **GPU に dispatch される最初の op** が NaN/Inf 発火点候補。

### 4.3 実装

`Models/Debug/ComputePlanInspector.swift` を新規追加し、`AudioSynthesizer.loadModels` の直後に呼べるユーティリティとする。出力は `Documents/Result/debug/compute_plan/<precision>_<computeUnits>.json`。

### 4.4 成果物

- `docs/2026-05-19/mlcomputeplan-dispatch-map.md`
  - 3 組合せの dispatch map 比較
  - GPU に行く op の列挙
  - Phase 3 で重点的に見るべき layer の候補

## 5. Phase 3 — Decoder 内部 divergence point 特定

### 5.1 アプローチ（両方を並行採用）

#### 5.1.1 PyTorch 参照値生成

- 入力 mel（mel_normalized）を固定し、PyTorch CPU F32 で Decoder forward を完全再現
- 中間 tensor を各境界で保存:
  - prenet 出力
  - positional encoding 加算後
  - self-attention pre / post
  - softmax pre / post
  - layer norm pre / post
  - FFN pre / post
  - mel_out / postnet_out
- スクリプト: `scripts/generate_decoder_reference.py`
- 出力: `data/2026-05-19/decoder_reference/<layer>_<step>.npy`

これにより CoreML 最終出力との diff から「どの step / どの量子化レベル で初めてズレるか」を絞る。

#### 5.1.2 多出力 CoreML モデル再変換

- `scripts/convert_decoder_multioutput.py` を新規追加
- 既存 Decoder の中間 tensor を `outputs=[...]` に追加した `Decoder_multiout.mlpackage` を生成
- F32 / F16 / Int8 の 3 種類
- iOS 側で読み込み、step 1 の各境界 tensor を `Documents/Result/debug/multiout/` に保存

### 5.2 比較軸

1. **F32 × cpuOnly（基準）vs F32 × cpuAndGPU 異常 run**: divergence する最初の layer = 原因 op の候補
2. **F32 × cpuAndGPU 異常 run vs 正常 run**: 同 engine 内の non-determinism の発火点
3. **F32 × cpuAndGPU vs F16 × cpuAndGPU**: precision 起因の境界
4. **CoreML 各 engine vs PyTorch 参照**: それぞれの絶対誤差の傾向

### 5.3 成果物

- `docs/2026-05-19/decoder-internal-divergence.md`
  - 各境界の diff 表（max_abs_diff / allclose 1e-2,1e-3,1e-4）
  - 初発火層の特定結論
  - Phase 4 dispatch map との対応付け
- `data/2026-05-19/decoder_reference/` と `decoder_multiout/`

## 6. Phase 5 — 実装方針の確定

Phase 1〜4 の結果を統合して以下を確定する。

### 6.1 12 組合せのカテゴリ分類

| カテゴリ | 定義 | 採用方針 |
|---|---|---|
| ① 1st call から正常 | 救済不要 | 本番デフォルト候補 |
| ② warm-up 1 回で正常 | dummy 1 回必要 | 本番採用可（起動 +1 推論コスト） |
| ③ warm-up 2 回で正常 | dummy 2 回必要 | 研究設定 |
| ④ 異常検出 + retry で正常 | 確率的、retry 必須 | 研究設定 |
| ⑤ computeUnits fallback | 別 engine で warm-up → 本番切替 | 研究設定 |
| ⑥ 安定化不能 | どの戦略でも `normal_loud` にならない | 非採用（記録のみ） |

### 6.2 実装提案項目

| 設定項目 | 通常利用デフォルト | 研究計測モード |
|---|---|---|
| precision | Phase 結果に依存 | UI で全選択可 |
| computeUnits | Phase 結果に依存 | UI で全選択可 |
| warm-up 回数 | カテゴリで決定 | 0 固定（観測） |
| 異常検出条件 | NaN/Inf、rms<3000、peak>32000 | フラグのみ、retry なし |
| retry 回数 | 最大 2 | 0 |
| fallback chain | 例: F16×cpuAndGPU → F32×cpuOnly | なし |
| ログ保存形式 | `Documents/Result/audio/<device>_<datetime>/run.json` | 既存 manual 退避形式 |

### 6.3 成果物

- `docs/2026-05-19/synthesis-stability-implementation-plan.md`
  - カテゴリ分類最終表
  - 実装提案コード（Swift 擬似コード）
  - 通常 / 研究モードの切替設計
  - 異常検出・retry・fallback の擬似コード

## 7. 進捗管理

タスクは Claude Code の TaskList で管理（タスク #1〜#6 が Phase 0〜5 に対応）。各 Phase の成果物 md ができた時点でそのタスクを completed にする。

## 8. 関連ドキュメント・コード

### 既存

- `docs/2026-05-17/fp32-cpuandgpu-quiet-vs-loud-investigation.md` — 直前の調査
- `ios/CoreMLAudioApp/CoreMLAudioApp/Models/Debug/DebugRunSnapshot.swift` — run スナップショット
- `ios/CoreMLAudioApp/CoreMLAudioAppTests/Fp32QuietInvestigationTests.swift` — 直接呼び出しテスト
- `scripts/compare_runs.py` — run 間 diff スクリプト
- `scripts/extract_manual_run.sh` — manual 操作の退避

### Phase 中に追加予定

- `scripts/aggregate_stability_matrix.py`（Phase 1）
- `ios/CoreMLAudioApp/CoreMLAudioApp/Models/Debug/ComputePlanInspector.swift`（Phase 4）
- `scripts/generate_decoder_reference.py`（Phase 3）
- `scripts/convert_decoder_multioutput.py`（Phase 3）
- `ios/CoreMLAudioApp/CoreMLAudioAppTests/StabilityMatrixTests.swift`（Phase 1〜2）

## 9. リスクと撤退ライン

| リスク | 対応 |
|---|---|
| MLComputePlan が期待する粒度の情報を返さない | 公開 API の制約を文書化し、PyTorch 参照 + 多出力モデルだけで Phase 3 を進める |
| Decoder 多出力モデル再変換で形状ずれが出る | PyTorch 参照のみで進める。多出力モデルは「取れる範囲で」のベストエフォート |
| 救済策 A〜I すべてが効かない組合せが出る | カテゴリ ⑥（安定化不能）として記録するだけ。研究結果としては「不能であること」も成果 |
| 計測が長時間化する | N=20 を N=10 に下げる、launch 系列を省く、を順に検討 |

「結論を急がない」方針なので、撤退は「これ以上やっても新情報が出ない」と判断できたときに限る。
