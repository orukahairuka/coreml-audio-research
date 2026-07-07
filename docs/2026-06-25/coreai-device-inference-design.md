# Core AI 実機推論パス追加の設計メモ

Phase 2 の目的：Core AI に変換した `.aimodel` を **iPhone 実機（iOS 27）で動かし、CPU / GPU / Neural Engine の違いを比較**する。
中心の問い：CoreML で観測した HiFi-GAN×NE の破綻（rms 約4.25倍→clip）が、別ランタイム Core AI でも再現するか＝**CoreML 固有か NE 実行系全般か**の切り分け。

このメモは実装前の設計整理。実装はこの後、最小構成から着手する。

---

## 1. 今できている `.aimodel` 一覧

`coreai_exports/`（gitignore。`scripts/convert_*_coreai.py` で生成）

| モデル | fp16 | fp32 | 入力 | 出力 |
|---|---|---|---|---|
| HiFi-GAN Generator | ✅ | ✅ | `mel (1,256,262)` | `audio (1,1,67072)` |
| Transformer Encoder | ✅ | ✅ | `mel (1,262,256)`, `pos (1,262)` | `memory (1,262,512)` |
| Transformer Decoder | ❌ | ✅ | `memory`, `decoder_input`, `pos` | `mel_out`, `postnet_out` |

- `.aimodel` は**フォルダ**（`metadata.json` + `main.mlirb` + `main.hash`）。関数名は `main`。
- fp16 Decoder が無いのは仕様上の問題ではない：**Core AI は精度を export 時に焼かず、実行時の compute unit で決まる**（§5）。fp32 Decoder を NE に dispatch すれば fp16 実行になる。mask 定数 `-2³²+1` の fp16 での扱いこそが実機で測りたい対象。
- 実機 PSNR 照合用の **PyTorch 参照入出力 npy** が `coreai_exports/reference/` に揃っている（`hifigan_mel_*`, `hifigan_pt_out_*`, `enc_*`, `dec_*`）。

---

## 2. iPhone 側で Core AI 推論パスを追加するのに必要なもの

### 2.1 依存・ビルド
- **Xcode 27 / iOS 27 必須**（`import CoreAI`）。実機 iPhone(3)@27.0 を使う。シミュレータは issue #49 で不可、Mac(26) も Python runtime 含め不可。
- アプリに **CoreAI.framework**（SDK 同梱）をリンク。必要に応じて `external/coreai-models/swift`（`CoreAISpeech` 等の runtime utilities）を SPM 依存に追加。ただし最小構成では **`import CoreAI` だけ**で足りる見込み（§4 の素の `AIModel` API を使う）。
- **`.aimodel` をアプリリソースとして同梱**。フォルダごと bundle に入れる（現状 `MLModels/` に `.mlpackage` を入れているのと同様、`AIModels/`（新設）に `.aimodel` を置く）。

### 2.2 新規追加するファイル（既存 MVVM 構成に沿う）
`ios/CoreMLAudioApp/CoreMLAudioApp/Models/Synthesis/` に Core AI 用 runner を追加：

| 新規ファイル | 役割 | 対応する既存 CoreML 版 |
|---|---|---|
| `CoreAIVocoderRunner.swift` | HiFi-GAN `.aimodel` を 1 回実行（mel→波形） | `VocoderRunner.swift` |
| `CoreAIModelLoader.swift` | `AIModel` を compute unit 指定でロード | `SynthesisViewModel` のロード部 |
| `CoreAIComputeUnit.swift` | cpu/gpu/neuralEngine の enum | `ComputeUnitOption.swift` |
| （後段）`CoreAIEncoderRunner` / `CoreAIDecoderRunner` | Transformer 段 | `EncoderRunner` / `DecoderRunner` |

記録・解析系（CSV / npy / timing / clipping）は**既存を流用**するので新規最小限（§7）。

---

## 3. CoreML 版の既存推論のどこを参考にするか

| 参考にする既存ファイル | 何を流用するか |
|---|---|
| `Models/Synthesis/VocoderRunner.swift` | mel の転置 + pad/crop（`[1,T,256]→[1,256,262]`）、predict 時間計測、contentT で波形を切り出すロジック。**Core AI 版でもそのまま使える前処理** |
| `Models/Synthesis/AudioSynthesizer.swift` | Encoder→Decoder ループ→HiFi-GAN のパイプライン orchestration |
| `Models/Synthesis/EncoderRunner` / `DecoderRunner` | Transformer 段の入出力の組み立て |
| `ViewModels/SynthesisViewModel.swift:146` | `MLModelConfiguration.computeUnits = selectedComputeUnit.mlComputeUnits` の適用箇所（Core AI では §5 の方式に置換） |
| `Models/Configuration/ComputeUnitOption.swift` | compute unit を UI 選択する enum の作り（Core AI 版の雛形） |
| `Models/Synthesis/VocoderStabilityTester.swift` | **最重要**。precision×shape×computeUnits を網羅実行し CSV（min/max/mean/rms/nan/inf/**ratio_over_099**=clip 指標）と波形 diff（npy）を吐く。Phase 2 の記録テンプレそのもの |
| `Models/Debug/ComputePlanInspector.swift` | CoreML の per-op device assignment（residency）を JSON 化。Core AI では `supportedComputeUnitKinds` 等で同等の可視化を作る（§7） |
| `Models/Synthesis/TimingInfo.swift` / `TimingJsonWriter.swift` | encoderMs/decoderMs/hifiganMs/RTF/modelSize の記録 schema |
| `Models/Audio/NpyWriter.swift` | 実機出力を npy 保存→オフライン PSNR/MCD |

---

## 4. Core AI 版の load / compile / run

SDK の swiftinterface とシンボル、`external/coreai-models/swift` の実例（`SpeechBundle.swift`, `speech-runner/SpeechRunnerMain.swift`）から確定した流れ：

```swift
import CoreAI

// --- Load（compute unit 指定は §5） ---
let model = try await AIModel(contentsOf: aimodelURL, options: specOptions)
guard let fn = try model.loadFunction(named: "main") else { /* error */ }

// --- 入力（NDArray） ---
//   mel: [1, 256, 262] float32 を NDArray に詰める
var mel = NDArray(shape: [1, 256, 262], scalarType: .float32)
var view = mel.mutableView(as: Float32.self)
// view に転置済み mel を書き込む（VocoderRunner の前処理を流用）

// --- Run ---
var out = NDArray(shape: [1, 1, 67072], scalarType: .float32)
var outputs = InferenceFunction.MutableViews()
outputs.insert(&out, for: "audio")
_ = try await fn.run(inputs: ["mel": mel],
                     states: InferenceFunction.MutableViews(),
                     outputViews: consume out)
let audio = out.view(as: Float32.self)   // [Float] へ
```

- **コンパイル**：初回ロード時に JIT、または事前に AOT。AOT は `xcrun coreai-build compile model.aimodel --platform iOS`、もしくは `AIModel.specialize(contentsOf:options:cache:cachePolicy:)`。**ANE 初回コンパイルが重い問題**（CoreML で 32–56 秒 or クラッシュを観測済み）が Core AI でも出るかは計測対象。
- 出力 descriptor は `model.functionDescriptor(for:"main")` で取得可（`SpeechRunnerMain` 参照）。出力名は変換時に付けた `audio` / `memory` / `mel_out` 等。

> 注：`NDArray` の具体的な init/view シグネチャと `fn.run` の正確な引数は実装時に SDK で最終確認する（上は SpeechRunner の用例ベース）。

---

## 5. compute unit（CPU / GPU / Neural Engine）の切り替え

**ここが Phase 2 の主役。** CoreML と仕組みが違う。

### CoreML（既存）
`MLModelConfiguration.computeUnits` に `.all / .cpuAndGPU / .cpuAndNeuralEngine / .cpuOnly` を設定。**組み合わせ**指定。

### Core AI（新）
`SpecializationOptions` に **単一の `ComputeUnitKind`** を `preferredComputeUnitKind` として渡し、それで `AIModel` をロードする：

```swift
// ComputeUnitKind: .cpu / .gpu / .neuralEngine（SDK シンボルで確認済み）
var opts = SpecializationOptions()           // 正確な init は実装時確認
opts.preferredComputeUnitKind = .neuralEngine
let model = try await AIModel(contentsOf: url, options: opts)
```

- 取りうる値：`ComputeUnitKind.cpu` / `.gpu` / `.neuralEngine`（+ `availableKinds`, アーキの `supportedComputeUnitKinds`）。
- **CoreML の組み合わせ（cpuAndGPU 等）と非対称**：Core AI は「優先する単一 unit」を指定する形。よって Phase 2 の比較軸は **cpu / gpu / neuralEngine の 3 値**にする。CoreML 側の cpuOnly/cpuAndGPU/cpuAndNE/all と完全一致はしないが、「NE に乗せたら壊れるか」という主問いには十分。
- 同じ fp32 `.aimodel` を 3 通りの compute unit でロードして回す＝精度（fp16 vs fp32）が **実行時に切り替わる**。これが「fp16 Decoder を別途作らない」理由（§1）。

---

## 6. 最小構成は HiFi-GAN だけにすべき

**結論：まず HiFi-GAN 単体でいく。** 理由：

- **破綻の震源そのもの**：ANE clipping は HiFi-GAN×NE で起きる（[[project_ane_stage_isolation]]）。主問いに最短で当たる。
- **単一モデル・単純 I/O**：mel(1,256,262)→波形、の 1 回 predict。自己回帰ループ不要。`VocoderRunner` の前処理をそのまま使える。
- **入力が用意済み**：`coreai_exports/reference/hifigan_mel_*.npy`（PyTorch 由来の mel）を実機に入れれば、Transformer 段を動かさずに HiFi-GAN だけ評価でき、**前処理差の混入を排除**できる（[[project_ios_mel_frontend_mismatch]] の教訓＝実機良 vs 壊れで切り分け）。
- **fp16 が変換済み**で素直。Decoder の fp16 mask 問題に最初から踏み込まなくていい。

Transformer（Encoder→自己回帰 Decoder）は段数・状態管理が増え、Decoder fp16 の論点も乗る。**HiFi-GAN で load/compile/run と記録の型を確立してから**全体へ広げる。

### MVP のスコープ
1. HiFi-GAN `.aimodel`（fp16 / fp32）をアプリに同梱
2. `reference/hifigan_mel_*.npy` を入力に、cpu / gpu / neuralEngine の 3 通りで実行
3. §7 の指標を記録、PyTorch 参照波形と PSNR 比較

---

## 7. 速度・音質・PSNR / MCD・clipping の記録

既存の記録インフラをほぼ流用できる。

| 指標 | 取り方 | 流用元 |
|---|---|---|
| **速度** | `fn.run` 前後を `CFAbsoluteTimeGetCurrent` で計測（predictMs）。初回 vs 2回目（コンパイル/ウォームアップ分離）、RTF | `VocoderRunner` の計測、`TimingInfo`（RTF/modelSize） |
| **clipping** | 波形の min/max/mean/**rms**/nan/inf/**ratio_over_0.99 / ratio_under_-0.99** を集計。CoreML の rms 約4.25倍膨張と同じ指標で並べる | `VocoderStabilityTester` の `WaveformStats` と CSV schema（`...,rms,nan_count,inf_count,ratio_over_099,ratio_under_minus_099,...`） |
| **PSNR** | 実機出力 npy vs PyTorch 参照 npy。合否基準（Apple skill）：**fp32 e2e >70dB / fp16 on-device >50dB**。40dB 未満は要調査 | `NpyWriter` で実機出力保存 →（新規）小さな PSNR 計算 or オフライン script |
| **MCD** | 実機出力 wav/npy をオフラインで処理 | `scripts/compare_mcd.py`（既存。on-device ではなく Python 後処理） |
| **residency** | どの op が NE / GPU / CPU に乗ったか。Core AI 版は `supportedComputeUnitKinds` と（あれば）コンパイル後の plan API で可視化 | `ComputePlanInspector`（CoreML の MLComputePlan 版）を Core AI 向けに作り直す |

- **出力先**は既存運用に合わせ `Documents/Result/`（CSV / npy / timing JSON）。
- **真値は npy**（wav は AudioPlayer 経由のみ更新される＝[[feedback_wav_vs_npy_gotcha]]）。

### 記録 CSV 案（HiFi-GAN MVP）
```
model,precision,compute_unit,run_index,status,predict_ms,
min,max,mean,rms,nan_count,inf_count,ratio_over_099,ratio_under_minus_099,
psnr_vs_pytorch
```
→ CoreML 版の成立マップに「Core AI」列として接続できる。

---

## 8. 未確定・リスク（実装時に潰す）

- `SpecializationOptions` の正確な init 構文、`NDArray` の init/view、`fn.run` の引数（`states` / `outputViews`）は SDK で最終確認。
- `.aimodel`（フォルダ）の bundle 同梱方法（フォルダ参照 vs リソースコピー）。
- ANE 初回コンパイルの所要時間・クラッシュ（CoreML で観測した 32–56 秒 or 落ちる現象）の Core AI 版での有無。
- `external/coreai-models/swift` を SPM 依存に足すか、`import CoreAI` だけで済ませるか（最小構成は後者狙い）。
- compute unit が「preferred」＝NE に乗らず CPU/GPU に落ちることがある（residency で要確認。CoreML の「compiles but runs on CPU」と同じ罠）。
- **Swift 経路（`CoreAIVocoderRunner` ほか）は未コンパイル**：`#if canImport(CoreAI)` ゲート下にあり、Mac(26) では CoreAI が import できず**一度もビルドされていない**。上記 SDK API（`NDArray` / `fn.run` / `consume` など）が実際のシグネチャと合うかは、**実機 iPhone@27 で初回ビルドが通るまで動作未保証**。最初のタスクはビルドを通すこと。
- **参照入力は実 mel を使う**：`reference/hifigan_mel_*.npy` は乱数ではなく実機 debug snapshot の postnet 出力（`--mel-npy` で焼き込み済み）。乱数入力だと HiFi-GAN 出力がほぼ無音（rms≈0.002）になり、CoreML で観測した実 mel でのクリップ破綻（rms 約4.25倍）と並べられないため。乱数フォールバック時の PSNR は「移植の数値等価性」検証止まりである点に注意。

---

## 9. 次アクション（最小実装）

1. ブランチはこの Core AI スコープ（`feature/coreai-conversion`）を継続 or 実機実装用に新ブランチ。
2. HiFi-GAN `.aimodel`（fp16/fp32）と `reference/hifigan_mel_*.npy` をアプリに同梱。
3. `CoreAIVocoderRunner` + `CoreAIComputeUnit` + 最小ロード/実行/記録を実装。
4. iPhone(3)@27.0 で cpu / gpu / neuralEngine の 3 通り実行 → CSV/npy 記録。
5. PyTorch 参照と PSNR、clipping 指標を CoreML 結果と並べる。

関連メモ：[[project_coreai_port]] / [[project_coreai_conversion_results]] / [[project_ane_clipping_finding]] / [[project_ane_stage_isolation]]
