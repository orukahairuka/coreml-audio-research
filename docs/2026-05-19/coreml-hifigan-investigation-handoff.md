# CoreML / HiFi-GAN 安定性調査 引き継ぎノート（2026-05-19）

ブランチ: `feature/fp32-gpu-memory-timing`
デバイス: iPhone 13（iOS 26.5）
日付: 2026-05-19

このドキュメントは `/clear` 後でも調査を再開できるように、ここまでの状態をまとめたもの。表現は抑えめにし、観測事実と推論を分けて記述している。

## 1. 背景

- もともとは「自動テストだと F32 × cpuAndGPU の音が出ない / quiet になる」という観察から始まった
- XCUITest 固有の問題と仮定して切り分けたが、その後 XCUITest や AVAudioSession 固有ではなく、CoreML の precision × computeUnits による数値挙動差として整理された
- 現在は precision × computeUnits の 12 組合せそれぞれが「正常な爆発音を出せるか」を体系的に調べるフェーズ

関連: [`all-engine-precision-stability-plan.md`](all-engine-precision-stability-plan.md)

## 2. 確定した事実（直接観測）

### 2.1 wav ファイルの扱い

- `audio/<archive>/output_<precision>_<computeUnit>.wav` は `AudioPlayer.play(baseName:)`（UI 操作経由）が呼ばれたときだけ書かれる
- Unit Test（Swift Testing の `Fp32QuietInvestigationTests` 等）は `AudioSynthesizer.synthesize()` を直接呼ぶだけで `AudioPlayer` を通らないため、テスト実行中に `output_*.wav` は更新されない
- 実際に 2026-05-19 のテスト実行で取得した `output_Float32_cpuAndGPU.wav` は 5/19 02:48 のタイムスタンプで、同日 21:21 開始のテストとは無関係の古いものだった
- テスト結果の真値は **debug snapshot の `waveform_postdeemph.npy`** にある
- `scripts/npy_to_wav.py` で npy → wav 変換が可能になっている

### 2.2 Phase 1（安定性マトリクス）の観測結果

実機 iPhone 13 で 3 precision × 4 computeUnits = 12 組合せの Repeat3（同一 `AudioSynthesizer` を 3 回呼ぶ）を計測（合計 14 テスト、所要 55 分）。

| precision \ computeUnit | cpuOnly | cpuAndGPU | cpuAndNE | all |
|---|---|---|---|---|
| Float32 | normal_loud | normal_loud（単発初回のみ確率的） | normal_loud | normal_loud |
| Float16 | normal_loud | normal_loud | clipped | clipped |
| Int8 | normal_loud | normal_loud | clipped | clipped |

- normal_loud の参考値: int16_rms ≈ 5000, peak ≈ 24000-28000
- clipped の値: F16×cpuAndNE は rms 13289 / peak 71468、Int8×cpuAndNE は rms 14154 / peak 87400
- iter1/2/3 はすべて bit-identical（同一インスタンス内）
- F32 × cpuAndGPU の単発初回（fresh AudioSynthesizer の 1 回目）は確率的で、5/19 セッションでは clipped (rms 8441 / peak 33162) を観測。同インスタンス 2 回目以降は normal_loud に収束
- F32 は warm 状態であれば 4 つの computeUnit いずれでも bit-identical に normal_loud（rms 5029 / peak 24326）

詳細: [`stability-matrix-analysis.md`](stability-matrix-analysis.md), [`stability-matrix-results.md`](stability-matrix-results.md), `data/2026-05-19/stability_matrix.csv`

### 2.3 Phase 4（MLComputePlan dispatch）の観測結果

iOS 17+ の `MLComputePlan` API で 18 plans を取得。実機計測した per-op device assignment:

- **F32 × cpuAndNE / all は NE dispatch されない**（F32 は ANE 非対応）。Phase 4 で NE op = 0 を確認
- F16 / Int8 × {cpuAndNE, all} は Decoder の 188 op と HiFi-GAN の 81 op が NE 行き
- HiFi-GAN F16 × cpuAndNE の NE 行き op 内訳: `add` 44 個（全部）、`conv` 17/74、`leaky_relu` 18/69、`conv_transpose` 1/4。最終 `tanh` は CPU

詳細: [`compute-plan-analysis.md`](compute-plan-analysis.md), [`mlcomputeplan-dispatch-map.md`](mlcomputeplan-dispatch-map.md)

### 2.4 HiFi-GAN 由来の振幅差を示唆する観測

Phase 1 の debug snapshot から `postnet_output`（HiFi-GAN 入力）と `waveform_predeemph`（HiFi-GAN 出力、de-emphasis 前）を比較:

| 設定 | postnet max | postnet mean | predeemph rms | predeemph max |
|---|---:|---:|---:|---:|
| F16 × cpuAndGPU | 0.7314 | 0.2694 | 0.0268 | 0.36 |
| F16 × cpuAndNE | 0.7363 | 0.2689 | 0.1140 | 0.96 |
| Int8 × cpuAndGPU | 0.7290 | 0.2652 | 0.0283 | 0.52 |
| Int8 × cpuAndNE | 0.7305 | 0.2645 | 0.1171 | 0.99 |

- Decoder 出力（postnet）は cpuAndGPU と cpuAndNE で min/max/mean がほぼ一致 → Decoder 由来の可能性は低い
- HiFi-GAN 出力（predeemph）の時点で振幅差が出ている → de-emphasis ではなく HiFi-GAN 内部で振幅差が生じている可能性が高い

### 2.5 PyTorch reference 比較

`scripts/generate_hifigan_reference.py` で PyTorch CPU F32 HiFi-GAN を実行（入力は iOS F16×cpuAndGPU run の postnet_output.npy）:

| 出力 | max | rms |
|---|---:|---:|
| PyTorch CPU F32 tanh_out | 0.3346 | 0.0269 |
| iOS F16 × cpuAndGPU predeemph | 0.36 | 0.0268 |
| iOS F16 × cpuAndNE predeemph | 0.96 | 0.1140 |

→ F16 × cpuAndGPU は PyTorch CPU F32 reference に近い値域、F16 × cpuAndNE は reference から外れる outlier。

## 3. 現時点の解釈（推論、断定はしない）

- clipping は Decoder 由来ではなく HiFi-GAN 内部で振幅差が生じている可能性が高い
- de-emphasis フィルタは主因ではないと考えられる
- residual connection に対応する `add` 44 op が全て NE 行きなので候補だが、`conv` / `leaky_relu` も一部 NE 行きなので op 単位の原因は未確定
- NE dispatch と clipping の相関は観測されたが、ANE 経路が直接の原因かは op 単位の中間出力比較がないと特定できない

## 4. 表現上の注意（次の人へ）

- 「ANE が悪い」とは書かない
- 「add が犯人」とは書かない
- 「F32 が ANE で正常に動いた」とは書かない（F32 は ANE 非対応で、computeUnits に NE を含めても CPU/GPU fallback する。これは絶対）
- 「HiFi-GAN の add が原因」とは書かない
- 「完全に解明した」「真犯人」「確定」などは使わない
- 観測（数値・sha・dispatch count）と推論（候補・可能性）を分けて記述する

## 5. 暫定的な実装方針

| 設定 | 評価 | 備考 |
|---|---|---|
| F16 × cpuAndGPU | 標準採用候補 | 既存本番設定、PyTorch reference に近い |
| F32 × cpuOnly | 採用候補 | manual baseline と一致 |
| F32 × cpuAndGPU | 条件付き採用候補 | warm-up 1 回入れた上で許可 |
| Int8 × cpuOnly | 採用候補 | manual baseline と近い |
| F16 / Int8 × cpuAndNE | 本番方針は未定 | clipping を観測。当面は UI に残す |
| F16 / Int8 × all | 本番方針は未定 | clipping を観測（F16/Int8 では all で NE 経路に乗る）。当面は UI に残す |
| ANE 経路全般 | 研究用 / 実験用として残す | 本番でどう扱うかは未定。方針が決まるまで UI から外さない |

これは Phase 5（実装方針の確定）で改めてレビューする。**方針が決まるまでは、どのセルも UI から外さず 12 セル全てを選択可能なまま残す。**

## 6. 次にやること

### Phase 2（mini）— 救済実験

これから着手する。以下を試して clipping が緩和できるか観測する:

- peak normalize（出力波形の peak を 1.0 に正規化）
- rms normalize（manual baseline の rms に合わせる）
- fixed gain 0.25 など固定ゲインを掛ける
- HiFi-GAN だけ cpuAndGPU に fallback して Decoder は cpuAndNE のまま動かす
- F16/Int8 × cpuAndNE で Repeat3 を取り、iter1/2/3 で値が変わるかを再確認

実装位置: `ios/CoreMLAudioApp/CoreMLAudioAppTests/Phase2RescueTests.swift` に戦略 A/B/C/F/G/H/I が既に書かれている。D/E や上記 mini は別途追加する。

### Phase 5 — 実装方針の確定

- §5 の表をベースに各設定の本番での扱いを検討（方針が決まるまで UI からは外さない）
- warm-up 戦略の本番組み込み（F32 × cpuAndGPU 用）
- 異常検出（NaN/Inf/rms/peak）+ retry の本番組み込み有無

### Op 単位の特定（重い作業、今は着手しない）

- HiFi-GAN を upsample stage で切った多出力 mlpackage を作る
- もしくは PyTorch 上で「特定 stage の出力を 2.5x スケール」などの摂動実験
- 着手するなら `upsample_3_resblock_2` 周辺（PyTorch 中間値で値域が大きい区間）に絞る

## 7. 重要なコード / docs / scripts の地図

### Swift（iOS）

- `ios/CoreMLAudioApp/CoreMLAudioApp/Models/Debug/DebugRunSnapshot.swift` — run snapshot 書き出し（既存）
- `ios/CoreMLAudioApp/CoreMLAudioApp/Models/Debug/ComputePlanInspector.swift` — MLComputePlan dispatch JSON 書き出し（新規）
- `ios/CoreMLAudioApp/CoreMLAudioAppTests/Fp32QuietInvestigationTests.swift` — Phase 1（12 組合せ Repeat3 ＋ directRun ＋ warmup test、14 個）
- `ios/CoreMLAudioApp/CoreMLAudioAppTests/Phase2RescueTests.swift` — Phase 2 救済実験（戦略 A/B/C/F/G/H/I、10 個）
- `ios/CoreMLAudioApp/CoreMLAudioAppTests/Phase4ComputePlanTests.swift` — Phase 4 dispatch 取得（Decoder / Encoder / HiFi-GAN ×組合せ、18 個）

### Python scripts

- `scripts/aggregate_stability_matrix.py` — Phase 1 集計（debug snapshot → CSV + md）
- `scripts/aggregate_compute_plan.py` — Phase 4 集計（compute_plan JSON → md）
- `scripts/generate_decoder_reference.py` — Decoder PyTorch reference（hook で 13 境界 capture）
- `scripts/compare_decoder_reference.py` — Decoder ref vs iOS run の diff
- `scripts/generate_hifigan_reference.py` — HiFi-GAN PyTorch reference（28 中間 tensor 保存）
- `scripts/npy_to_wav.py` — debug snapshot の waveform_postdeemph.npy を WAV 化

### docs（2026-05-19）

- `docs/2026-05-19/all-engine-precision-stability-plan.md` — Phase 0-5 全体計画
- `docs/2026-05-19/stability-matrix-results.md` — Phase 1 生データ集計（auto 生成）
- `docs/2026-05-19/stability-matrix-analysis.md` — Phase 1 観測整理
- `docs/2026-05-19/mlcomputeplan-dispatch-map.md` — Phase 4 生データ集計（auto 生成）
- `docs/2026-05-19/compute-plan-analysis.md` — Phase 4 観測整理
- `docs/2026-05-19/coreml-hifigan-investigation-handoff.md` — 本ドキュメント

### data

- `data/2026-05-19/stability_matrix.csv` — Phase 1 CSV
- `data/2026-05-19/hifigan_reference/f16GpuRef/*.npy` — PyTorch HiFi-GAN reference 中間 tensor

### iOS run archive

- `audio/iPhone_3_phase1_20260519/debug/` — Phase 1 の debug snapshot（78 run、5/17 過去 run も混在）
- `audio/iPhone_3_phase1_20260519/playable/` — npy_to_wav で再生可能化した wav 群（15 個）
- `audio/iPhone_3_phase4_20260519/`, `audio/iPhone_3_phase4_20260519_round2/` — Phase 4 JSON

## 8. /clear 後の再開用プロンプト

> CoreML / HiFi-GAN の precision × computeUnits の安定性調査を継続したい。
> ブランチは `feature/fp32-gpu-memory-timing`。
> 引き継ぎノートは `docs/2026-05-19/coreml-hifigan-investigation-handoff.md` にある。
>
> 次にやりたいのは「Phase 2 mini の救済実験」。
> 具体的には:
> - F16/Int8 × cpuAndNE の clipping を peak normalize / rms normalize / fixed gain で緩和できるか
> - HiFi-GAN だけ cpuAndGPU に fallback して Decoder は cpuAndNE のまま動かす案の検証
> - F16/Int8 × cpuAndNE で Repeat3 が iter1/2/3 同値かを再確認
>
> 表現方針: 「重大発見」「真犯人」「確定」「バグ確定」などの強い表現は避け、観測と推論を分けて書くこと。
> F32 が ANE で動く可能性は最初から無いので、その方向の解釈は書かないこと。
