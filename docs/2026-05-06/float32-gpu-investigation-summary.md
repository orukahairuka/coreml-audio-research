# `feature/float32-gpu-investigation` ブランチまとめ

`main` から 21 コミット先行している `feature/float32-gpu-investigation` ブランチで
やったこと（計測機構整備 → 12 通りベースライン取得 → Float32 × GPU 飽和の原因調査
→ HiFi-GAN shape mode 検討 → 実機 GPU 経路の E5RT 回避策確定）の総まとめ。
PR は [#15](https://github.com/orukahairuka/coreml-audio-research/pull/15) で立てた。

## PR 本文

### タイトル

```
計測機構整備と HiFi-GAN fixed262 化で実機 GPU 経路を安定化
```

### 本文（Summary / Test plan）

```markdown
## Summary

- CoreML 各段の `predict()` 所要時間・出力 mel・出力 wav・モデルサイズ・RTF を
  自動保存する計測機構を入れ、12 通り (Float32/Float16/Int8 × cpuOnly/cpuAndGPU/cpuAndNE/all)
  をワンバッチで取れるようにした (`docs/2026-04-25/timing-measurement.md`,
  `docs/2026-04-27/quantization-pareto-baseline.md`)。
- Float32 × GPU で HiFi-GAN 出力が DC 飽和する件を切り分け、
  `allowLowPrecisionAccumulationOnGPU = false` では直らないこと（仮説 1 否定）と、
  「中間テンソル fp16 保持」仮説が公式 Typed Execution と整合しないこと
  （仮説 2 撤回）を記録 (`docs/2026-04-27/float32-gpu-accumulation-experiment.md`)。
  原因は未特定で、追加調査の方針も同ドキュメントに残した。
- 実機テストで RangeDim を広く取った HiFi-GAN が GPU 経路で
  `E5RT: No memory object bound to port` で落ちることを確認し、
  HiFi-GAN を `fixed262` (1, 256, 262) へ寄せた本番デフォルトに切替。
  Transformer Encoder/Decoder も同じ shape mode を持つよう揃え、
  `ShapeModeOption` で UI から 4 通り (fixed262 / range16-384 / range16-1000 / range1-1000)
  を切替可能にした。
- 上記の検証用に `VocoderStabilityTester`（PyTorch / CoreML CPU/GPU/ALL の出力差分テスト）と
  `scripts/compare_hifigan_paths.py`、`scripts/convert_hifigan.py --all-variants`
  （7 通り一括生成）、Int8 安定性テストを追加。

## Test plan

- [ ] `PronounSE/venv/bin/python scripts/aggregate_metrics.py` で 12 通りの集計が出る
- [ ] iOS 実機 (iPhone) で fixed262 + Float16 + cpuAndGPU の合成が成功する
- [ ] iOS 実機で fixed262 + Float32 の cpuOnly / cpuAndNE 合成が成功する
- [ ] HiFi-GAN 安定性テスト (`VocoderStabilityTester`) を UI から走らせて
      Float16 fixed262 cpuAndGPU/all の出力差分が許容範囲内
- [ ] `scripts/convert_hifigan.py --all-variants` で 7 通りの mlpackage が生成できる
```

## ブランチに含まれる作業（コミット順）

下から上が時系列。

| コミット | 区分 | 内容 |
|---|---|---|
| `0878a53` | 計測 | CoreML `predict()` 所要時間を計測 |
| `8bfbfe7` | 計測 | 合成成功時に Timing 情報を JSON で自動保存 |
| `b423316` | 計測 | `extract_ui_test_results.sh` を Documents/Result/ 全体取得に変更 |
| `bad47af` | docs | タイミング計測機構の実装メモ (`docs/2026-04-25/timing-measurement.md`) |
| `c333917` | docs | README に `ui-test-loop-fixes` 索引追加 |
| `dc60cd9` | 計測 | 出力 wav のファイル名を `Precision_ComputeUnit` ラベル形式に |
| `d07aa3a` | 計測 | `TimingInfo` にモデルサイズと Real-time factor を追加 |
| `8164d04` | 計測 | 12 通りの計測値を集計するスクリプトを追加 |
| `61f76f5` | 計測 | `extract_ui_test_results.sh`: コンテナ判定で `mel/` と `timing/` の両方の存在を要求 |
| `4bbb0d0` | docs | 量子化 × デバイス 12 通りベースライン (`docs/2026-04-27/quantization-pareto-baseline.md`) |
| `b81150d` | docs | Float32 × GPU 飽和の累積精度切り分け実験 (`docs/2026-04-27/float32-gpu-accumulation-experiment.md`) |
| `ed17f09` | 検証 | `scripts/compare_hifigan_paths.py`（PyTorch / CoreML CPU/GPU/ALL の HiFi-GAN 出力比較） |
| `d69f3df` | 変換 | HiFi-GAN 変換時の `RangeDim` 下限を 1 → 16 に上げる |
| `9285e8e` | 変換 | `convert_hifigan.py` に `--shape-mode` と `--all-variants`（7 通り一括生成） |
| `af82321` | iOS | `VocoderStabilityTester` と Float16 fixed262 cpuAndGPU/all 出力差分テスト追加 |
| `3ea8151` | iOS | 本番 HiFi-GAN を `fixed262` デフォルトに切替し、UI から shape mode を選択可能に |
| `c511df2` | iOS | `ContentView` を `ScrollView` で包む。Picker のヒット領域明示 |
| `091f5cc` | iOS | HiFi-GAN 安定性テストに Int8 (legacy `RangeDim` 16-1000) を追加 |
| `325618b` | mlpackage | `range1` HiFi-GAN バリアント追加 + Int8 Transformer 再生成 |
| `89a7726` | iOS / 変換 | Transformer Encoder/Decoder にも shape mode を導入し fixed262 対応 |

## 調査内容まとめ

### 1. 計測機構（Phase 1 ベースライン取得の土台）

`docs/2026-04-25/timing-measurement.md` に詳細。

- `EncoderRunner` / `DecoderRunner` / `VocoderRunner` の `predict()` 実時間を取り、
  `TimingInfo` 構造体に encoder ms / decoder 全ステップ ms / decoder/step / hifigan ms /
  total ms / RTF / 3 モデル合計サイズ MB を載せて JSON 保存。
- 出力 wav は `output_<Precision>_<ComputeUnit>.wav` の命名で 12 通り並列保存。
- `XCUITest` (`testCaptureAllCombinations`) でアプリを 12 回操作し、
  `Documents/Result/{mel,timing,*.wav}` をホストに `extract_ui_test_results.sh`
  で吸い出し、`scripts/aggregate_metrics.py` で 1 表に集計。

### 2. 量子化 Pareto ベースライン（シミュレータ）

`docs/2026-04-27/quantization-pareto-baseline.md` に詳細。

- サイズは Float32 181.5 MB → Float16 90.9 MB → Int8 45.9 MB（ほぼ半分ずつ）。
- シミュレータでは全組み合わせ RTF 2.5–4.4 でリアルタイム未達。実機 ANE は別計測。
- mel L1: Float16 0.62–0.84、Int8 2.6–2.94。Cos 類似度はどれも 0.997 以上。
- **ただしシミュレータの Float32 × GPU 経路だけ DC 1.0 飽和**。これが次の調査に繋がる。

### 3. Float32 × GPU 出力飽和の原因調査（**未特定**）

`docs/2026-04-27/float32-gpu-accumulation-experiment.md` に詳細。

- 観察: Float32 × cpuAndGPU / all で出力 peak/rms が 32767（int16 飽和）に張り付く。
  Float16 / Int8 は全 computeUnit で正常。
- Decoder 出力 mel は GPU/CPU で完全一致 → **破綻は HiFi-GAN 段に局所化**。
- 仮説 (1) 「行列積の累積が fp16」: `MLModelConfiguration.allowLowPrecisionAccumulationOnGPU = false`
  を付けて再計測 → peak/rms が完全一致したため **否定**。
- 仮説 (2) 「中間テンソル保持精度が fp16」: Apple 公式 Typed Execution の
  「ML programs … all variables in the program are strongly typed」「`compute_precision=FLOAT32`
  のモデルは guaranteed to run with float 32 precision on all hardware and software versions」
  と整合しないため **撤回**。
- **結論: 原因は本実験では特定できていない**。同ドキュメント末尾に追加調査の軸
  （MIL の型確認 / op レベル切り分け / 実機 vs シミュレータ / `minimum_deployment_target` /
  既知 issue / 中間値の数値追跡）を残した。

### 4. 実機テストで判明した HiFi-GAN RangeDim 上限と GPU 経路の E5RT

実機 (iPhone) で動作確認したところ、HiFi-GAN を広い `RangeDim` で書き出したモデル
（`RangeDim(*, 1000)`）を GPU 経路で実行すると
`E5RT: No memory object bound to port` で失敗することを確認。

`ShapeModeOption.swift` に 4 つの shape mode を定義して切替可能にした:

| shape mode | 入力 shape | 用途 |
|---|---|---|
| `fixed262` | (1, 256, 262) | **本番推奨**。実機で Float32/Float16 とも全 computeUnits 成功 |
| `range16_384` | `RangeDim(16, 384, default=262)` | GPU 経路でも動作する可変長候補 |
| `range16` | `RangeDim(16, 1000, default=100)` | GPU 経路で E5RT 失敗（再現用） |
| `range1` | `RangeDim(1, 1000, default=100)` | legacy 命名モデルへのフォールバック |

→ **本番デフォルトを `fixed262` に切替**。Transformer Encoder/Decoder にも同じ
`ShapeModeOption` を導入して shape を揃え、UI（`ContentView` の Picker）から 4 通り選択可。

### 5. HiFi-GAN 安定性テスト機能

`VocoderStabilityTester.swift`（約 600 行）を追加。UI から起動して以下を回す:

- **PyTorch / CoreML CPU/GPU/ALL** の HiFi-GAN 出力比較
  （対応する Python 側スクリプトは `scripts/compare_hifigan_paths.py`）
- **Float16 fixed262 cpuAndGPU/all** の出力差分テスト
- **Int8 (legacy `RangeDim` 16-1000)** の安定性テスト

### 6. 変換スクリプト整備

- `scripts/convert_hifigan.py`: `--shape-mode` 指定 + `--all-variants` で 7 通り
  （Float32/Float16/Int8 × shape mode の組み合わせ）を一括生成。
- `scripts/convert_transformer.py`: shape-mode 対応で `fixed262` 版も生成。
- `RangeDim` 下限を 1 → 16 に引き上げ（変換時のショートシーケンス問題回避）。

## 残課題（次ブランチ候補）

- Float32 × GPU 飽和の **原因特定**: MIL の型確認、op レベル切り分け、
  実機 vs シミュレータの再確認、`minimum_deployment_target` の影響
  （`docs/2026-04-27/float32-gpu-accumulation-experiment.md` の「今後の調査方針」に列挙済み）。
- 実機 (iPhone) での 12 通り再計測（シミュレータ値は研究データに使えない）。
- 複数音声入力・複数試行で計測値の安定性確認。
- 波形ドメインの音質指標（SNR / MCD / PESQ / STOI）。
- メモリピーク計測（Instruments Allocations もしくは `os_proc_available_memory`）。

## 関連ドキュメント

- 計測機構: [`docs/2026-04-25/timing-measurement.md`](../2026-04-25/timing-measurement.md)
- UI テスト修正: [`docs/2026-04-25/ui-test-loop-fixes.md`](../2026-04-25/ui-test-loop-fixes.md)
- 12 通りベースライン: [`docs/2026-04-27/quantization-pareto-baseline.md`](../2026-04-27/quantization-pareto-baseline.md)
- Float32 × GPU 切り分け: [`docs/2026-04-27/float32-gpu-accumulation-experiment.md`](../2026-04-27/float32-gpu-accumulation-experiment.md)
- 起点となった初期報告: [`docs/2026-04-05/float32-gpu-debug-report.md`](../2026-04-05/float32-gpu-debug-report.md)
- 研究計画: [`docs/2026-04-22/research-plan.md`](../2026-04-22/research-plan.md)
