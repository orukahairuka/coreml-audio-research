# CoreML predict() 所要時間の計測機構を追加

`feature/timing-measurement` ブランチで実装した内容のまとめ。

## 目的

精度 (Float32 / Float16 / Int8) × 計算デバイス (cpuOnly / cpuAndGPU / cpuAndNE / all) の 12 通りで合成を回したときに、CoreML 各段の `predict()` がどれくらい時間を食っているかを定量的に取れるようにする。

研究計画 [research-plan](../2026-04-22/research-plan.md) の Phase 1（量子化パレート）に必要なベースラインデータの取得手段。

## スコープ

- **計測対象**: `model.prediction(from:)` の呼び出し時間のみ
  - 入力 `MLMultiArray` 構築や `[Float]` への展開などの前後処理は除外
  - メル抽出やデエンファシスなどの非 CoreML 処理も除外
- **計測しないもの**: モデルロード時間、メル抽出、後処理、UI レンダ時間
- **計測タイミング**: アプリで合成を実行するたびに自動記録
- **保存先**: `Documents/Result/timing/timing_<Precision>_<ComputeUnit>.json`

## 実装

### 新規追加

| ファイル | 役割 |
|---|---|
| `Models/Synthesis/TimingInfo.swift` | encoder / decoder (total + step count) / hifigan の所要時間を保持する `Codable` struct |
| `Models/Synthesis/TimingJsonWriter.swift` | `TimingInfo` を `Documents/Result/timing/timing_<Precision>_<ComputeUnit>.json` に書き出す |

### 変更

| ファイル | 変更点 |
|---|---|
| `Models/Synthesis/EncoderRunner.swift` | `run()` の戻り値を `(memory, predictMs)` のタプルに変更。`predict()` の前後を `CFAbsoluteTimeGetCurrent()` で挟む |
| `Models/Synthesis/DecoderRunner.swift` | 同様に戻り値に `totalPredictMs` を追加。自己回帰ループで `predict()` ごとに加算 |
| `Models/Synthesis/VocoderRunner.swift` | 同様に戻り値を `(waveform, predictMs)` に変更 |
| `Models/Synthesis/AudioSynthesizer.swift` | 各 Runner から返ってきた時間を集約して `TimingInfo` を作り、`SynthesisResult.timing` に詰める |
| `Models/Synthesis/SynthesisResult.swift` | `timing: TimingInfo` を追加 |
| `ViewModels/SynthesisViewModel.swift` | 合成成功時に `saveTimingArtifact(result:)` を呼んで JSON を書き出す |
| `scripts/extract_ui_test_results.sh` | 取り出し対象を `Documents/Result/mel/` から `Documents/Result/` 全体に拡大（mel と timing をまとめて取得） |

## 出力フォーマット

`Documents/Result/timing/timing_Float16_cpuAndGPU.json` の例:

```json
{
  "computeUnit": "cpuAndGPU",
  "decoderAvgPerStepMs": 17.2,
  "decoderStepCount": 262,
  "decoderTotalMs": 4506.4,
  "encoderMs": 12.3,
  "hifiganMs": 89.5,
  "precision": "Float16",
  "totalPredictMs": 4608.2
}
```

| キー | 内容 |
|---|---|
| `precision` | `ModelPrecision.rawValue` (Float32 / Float16 / Int8) |
| `computeUnit` | `ComputeUnitOption.rawValue` (cpuOnly / cpuAndGPU / cpuAndNE / all) |
| `encoderMs` | Encoder の `predict()` 1 回ぶん (ms) |
| `decoderTotalMs` | Decoder の `predict()` 全ステップ合計 (ms) |
| `decoderStepCount` | Decoder のステップ回数 (= 入力フレーム数) |
| `decoderAvgPerStepMs` | `decoderTotalMs / decoderStepCount` |
| `hifiganMs` | HiFi-GAN の `predict()` 1 回ぶん (ms) |
| `totalPredictMs` | encoder + decoder + hifigan の合計 |

## 使い方

1. シミュレータを起動（クローンが作られないよう事前 boot 推奨）:
   ```bash
   xcrun simctl boot 7B5CCAF6-0BCB-4BE4-9929-BD6FA849B5BF  # iPhone 17 Pro / iOS 26.2
   ```

2. XCUITest を実行（mel と timing の両方が 12 通りぶん書き出される）:
   ```bash
   cd ios/CoreMLAudioApp
   xcodebuild -project CoreMLAudioApp.xcodeproj \
     -scheme CoreMLAudioApp \
     -destination 'platform=iOS Simulator,id=7B5CCAF6-0BCB-4BE4-9929-BD6FA849B5BF' \
     -parallel-testing-enabled NO \
     -only-testing:CoreMLAudioAppUITests/CoreMLAudioAppUITests/testCaptureAllCombinations \
     test
   ```

3. シミュレータから成果物を取り出す:
   ```bash
   ./scripts/extract_ui_test_results.sh
   ```
   `result/mel/` と `result/timing/` の両方が更新される。

4. 集計（Python）:
   ```python
   import json, glob
   for path in sorted(glob.glob('result/timing/*.json')):
       with open(path) as f:
           t = json.load(f)
       print(f"{t['precision']}/{t['computeUnit']}: total={t['totalPredictMs']:.1f}ms")
   ```

## 制約・注意

- **シミュレータの計測値は研究データには使えない**
  - ANE は実在せず CPU フォールバック（cpuAndNE = cpuOnly になる、メル比較で確認済み）
  - GPU は Mac の Metal で iPhone 実機の GPU と挙動が違う
  - シミュレータの数字は「相対比較・計測機構の動作確認」用途に限る
- **実機計測**は別タスク。実機接続 + Instruments の Core ML テンプレートとの併用が望ましい
- `playOutput()` 等の前後処理は計測に含まれていないので「ボタン押してから音が鳴るまで」の体感時間とは一致しない

## ブランチ・コミット

`feature/timing-measurement` を main から切って作業:

```
8bfbfe7 合成成功時に Timing 情報を JSON で自動保存する
0878a53 CoreML predict() の所要時間を計測する
```

`feature/ui-test-combinations` のマージ後（main = aae9627 時点）にリベース済み。
