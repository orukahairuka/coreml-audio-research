# Float32 × GPU 出力飽和の原因切り分け（途中経過・原因未特定）

> **ステータス**: 当初の2仮説のうち (1) は否定、(2) は Apple 公式 Typed Execution
> ドキュメントの記述と矛盾するため断定不可。**原因は本実験では特定できていない**。
> 一から再調査が必要（[今後の調査方針](#今後の調査方針) 参照）。

[`float32-gpu-debug-report`](../2026-04-05/float32-gpu-debug-report.md) で記録した
「Float32 モデルを GPU 経路で動かすと出力が壊れる」現象について、原因を
**累積 (accumulation) 精度**か **中間テンソル保持精度** かのどちらに帰すか
を切り分けようとした検証メモ。当初の仮説立てと実験記録は残しつつ、
公式仕様と矛盾する結論部分は撤回する。

## 起点となった観察

シミュレータで 12 通りバッチ取得した wav の振幅を比較したところ、
Float32 の GPU 経路だけが全サンプル飽和していた。

| 組み合わせ | peak | rms |
|---|---:|---:|
| Float32 / cpuOnly | 24327 | 5029 |
| Float32 / cpuAndNE | 24327 | 5029 |
| **Float32 / cpuAndGPU** | **32767** | **32679** |
| **Float32 / all** | **32767** | **32679** |
| Float16 / 全パターン | 25000〜28000 | 5000 前後 |
| Int8 / 全パターン | 25000〜28000 | 5500 前後 |

- peak=32767 は int16 の最大値。rms ≈ peak は **全サンプルが最大値に張り付いた DC 飽和**を意味する。
- これは fp32 → int16 化のときに `Inf`／非常に大きな値がクランプされた典型パターン。

さらに、Decoder 出力 (HiFi-GAN への入力) の mel.npy を 4 通り全部比較すると
Float32 では `min=-80, max=-6.85, mean=-53.02` で **完全一致**。
つまり Encoder/Decoder は GPU でも壊れず、**HiFi-GAN を GPU で走らせた時だけ**
出力が破綻していると確定した。

## 当初立てた仮説（後述のとおり (2) は撤回）

「CoreML の GPU バックエンド (Metal Performance Shaders) で Float32 重みのモデルを
動かすとき、内部演算は fp16 で行われる」という前提で、破綻点として2つを想定した:

1. **行列積の累積 (accumulator) が fp16** — 畳み込みで多数の積を足し合わせる
   ときに途中で fp16 範囲 (±65504) を超えて Inf になる
2. **中間テンソルの保持精度が fp16** — 演算結果を fp16 で次レイヤに渡すため、
   一度でも活性が範囲を超えると Inf が伝播する

(1) なら CoreML の `MLModelConfiguration.allowLowPrecisionAccumulationOnGPU = false`
で fp32 累積を強制でき、直る。
(2) なら同フラグでは直らない（中間ストレージは別レイヤの設定）。

> **注**: 仮説 (2) の前提（「GPU が float32 重みでも中間を fp16 で保持する」）は、
> 後で確認した Apple 公式 Typed Execution ドキュメントの記述と整合しない。
> 同ドキュメントは "ML programs ... all variables in the program are strongly typed"、
> "The runtime respects the explicit types as the minimum precision, and will not
> reduce the precision"、`compute_precision=FLOAT32` のモデルは "guaranteed to run
> with float 32 precision on all hardware and software versions" と明記している。
> したがって仮説 (2) は **公式仕様レベルでは成立しない前提に基づいていた**。

## 実験

`AudioSynthesizer.loadModels` で 3 モデルロード時の config に
`allowLowPrecisionAccumulationOnGPU = false` を付け、
`testCaptureAllCombinations` を再実行した。

```swift
let config = MLModelConfiguration()
config.computeUnits = computeUnits
config.allowLowPrecisionAccumulationOnGPU = false  // 検証用に追加
```

## 結果

| 組み合わせ | フラグ前 peak | フラグ後 peak | フラグ前 rms | フラグ後 rms |
|---|---:|---:|---:|---:|
| Float32 / cpuAndGPU | 32767 | **32767** | 32678.8 | **32678.8** |
| Float32 / all       | 32767 | **32767** | 32678.8 | **32678.8** |

**完全に同じ値**。フラグは効かなかった。

## 結論（暫定・原因未特定）

- **仮説 (1) は否定**: `allowLowPrecisionAccumulationOnGPU = false` を付けても
  出力 peak / rms が完全に一致したため、「行列積の累積精度」が原因ではない。
- **仮説 (2) は撤回**: 上記注のとおり Apple 公式 Typed Execution と整合しない。
  「GPU が中間を勝手に fp16 化する」という説明は公式仕様レベルでは成立しない。
- **原因は本実験では特定できていない**。確定しているのは観察事実のみ:

| 精度 | CPU | NE | GPU | all |
|---|:-:|:-:|:-:|:-:|
| Float32 | ✅ | ✅ | ❌ DC 飽和 | ❌ DC 飽和 |
| Float16 | ✅ | ✅ | ✅ | ✅ |
| Int8    | ✅ | ✅ | ✅ | ✅ |

加えて、Decoder 出力 mel は GPU/CPU で完全一致しており、**破綻が HiFi-GAN 段に
局所化されている** ことも確定している。ただし「なぜ HiFi-GAN だけか」を
メカニズムとして説明する根拠は本実験には無い。

## 実用上の指針（観察事実ベース）

原因が未特定でも、観察された症状からくる回避策は変わらない:

- **Float32 を使う計測は CPU / NE のみ**。GPU/all は計測対象から外すか、
  「破綻する」事実そのものを記録する。
- GPU を使いたい場合は Float16 か Int8 にフォールバック。
- 調査用に追加した `allowLowPrecisionAccumulationOnGPU = false` は効果がないため
  本実験後にコードから削除（diff 残さない）。

## 補足: なぜ HiFi-GAN だけか（仮説、未検証）

Encoder/Decoder（Transformer 系・LayerNorm と softmax で活性が正規化される）と
比較して、HiFi-GAN は転置畳み込みでアップサンプリングする構造のため
活性の動的範囲が大きいモデルである。Decoder 出力 mel が GPU/CPU で完全一致した
観察と、HiFi-GAN 段で破綻が起きていることは整合的だが、
**「動的範囲の広さが直接の原因」と断定する根拠は本実験には無い**。
原因の特定は次節の追加調査による。

## 今後の調査方針

公式仕様（float32 typed の `mlprogram` は GPU でも float32 精度で実行されるはず）
と観察結果（GPU 経路で HiFi-GAN が破綻する）の不一致を解消するため、
原因を一から切り分け直す。検討中の調査軸:

1. **変換結果の型を確認**: 生成された `.mlpackage` の MIL を coremltools で開き、
   HiFi-GAN の各 op が本当に float32 で型付けされているかを確認する。
   （`compute_precision=FLOAT32` 指定が op 単位で正しく伝播しているかの検証）
2. **op レベル切り分け**: HiFi-GAN を細分化（前半／後半、各 ResBlock、
   転置畳み込み単独）して GPU 経路で実行し、どの op で出力が壊れ始めるかを特定。
3. **実機 vs シミュレータ**: 現在の観察は iOS シミュレータ上のもの。
   実機（macOS ホスト or iOS 実機）でも同じ症状が出るかを確認し、
   シミュレータ GPU 経路固有の問題でないかを切り分ける。
4. **`minimum_deployment_target` の影響**: 変換時にターゲット OS バージョンを
   明示していない場合、古い Core ML ランタイム向けの ML Program が生成され
   typed execution の挙動が変わる可能性がある。新しいターゲットで再変換して比較。
5. **既知 issue の確認**: coremltools / Core ML の GitHub issue で、typed execution
   + GPU + 転置畳み込み（または HiFi-GAN 系モデル）に関する報告がないか確認。
6. **数値の追跡**: 中間出力を取り出せるよう Decoder ↔ HiFi-GAN の境界を分割し、
   実際に GPU 上のどの段で値域が破綻するかを数値ログで追う。
