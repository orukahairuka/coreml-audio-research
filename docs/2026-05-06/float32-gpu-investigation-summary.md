
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

### 3. Float32 × GPU 出力飽和の原因調査（**RangeDim 起因の可能性大 / 未検証**）

`docs/2026-04-27/float32-gpu-accumulation-experiment.md` に詳細。

#### 当初の調査（広い RangeDim HiFi-GAN を使用）

- 観察: Float32 × cpuAndGPU / all で出力 peak/rms が 32767（int16 飽和）に張り付く。
  Float16 / Int8 は全 computeUnit で正常。
- Decoder 出力 mel は GPU/CPU で完全一致 → **破綻は HiFi-GAN 段に局所化**。
- 仮説 (1) 「行列積の累積が fp16」: `MLModelConfiguration.allowLowPrecisionAccumulationOnGPU = false`
  を付けて再計測 → peak/rms が完全一致したため **否定**。
- 仮説 (2) 「中間テンソル保持精度が fp16」: Apple 公式 Typed Execution の
  「ML programs … all variables in the program are strongly typed」「`compute_precision=FLOAT32`
  のモデルは guaranteed to run with float 32 precision on all hardware and software versions」
  と整合しないため **撤回**。
- この時点での結論: 「原因は本実験では特定できていない」。

#### その後浮上した有力仮説: そもそも RangeDim + GPU の問題だった可能性

時系列を並べると、当時の HiFi-GAN は広い `RangeDim`（1〜1000 または 16〜1000）で
書き出されていた。その後 section 4 で実機テストを行ったところ、**まったく同じ
「広い RangeDim + GPU 経路」で `E5RT: No memory object bound to port` 失敗**することが判明し、
`fixed262`（1, 256, 262）への切替で Float32 / Float16 とも全 computeUnits で成功している。

つまり、当初「Float32 × GPU の問題」として切り分けていた症状は、実際には
**「広い RangeDim + GPU 経路」という同一の根本原因**が、シミュレータでは E5RT で落ちずに
DC 飽和という別症状で表に出ていただけ、という見方が自然になる。Float32 だけ壊れて
Float16 / Int8 は正常だった点は、そもそも GPU 経路で Float16/Int8 は内部精度が違うため
症状の出方が変わったという解釈で十分整合する。

**未検証**。検証は単純で、`fixed262` HiFi-GAN でシミュレータの 12 通りを再計測し、
Float32 × cpuAndGPU / all の peak/rms が正常範囲（Float32 / cpuOnly と同じ ~24327 / ~5029）
に戻れば、独立した「Float32 × GPU バグ」は存在せず、原因は RangeDim だった、と結論できる。
逆に依然として飽和するなら独立問題として再調査の余地が残る。

#### 元ドキュメントに残した追加調査の軸（fixed262 再計測で消えなかった場合に走る）

- `.mlpackage` の MIL を coremltools で開いて HiFi-GAN の op が float32 で型付けされているか
- HiFi-GAN を細分化（前半／後半、各 ResBlock、転置畳み込み単独）して GPU で破綻 op を特定
- 実機 vs シミュレータ
- `minimum_deployment_target` の影響
- coremltools / Core ML の既知 issue
- Decoder ↔ HiFi-GAN 境界を分割して中間値の数値追跡

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

- **`fixed262` HiFi-GAN でシミュレータ 12 通り再計測**（最優先）— Float32 × GPU 飽和が
  消えれば「RangeDim 起因」と確定し、section 3 の調査軸はクローズできる。
- 実機 (iPhone) での 12 通り計測（シミュレータ値は研究データに使えない）。
- 上で消えなかった場合のみ、Float32 × GPU 飽和の op レベル切り分け。
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
