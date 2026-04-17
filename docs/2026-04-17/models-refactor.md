# セッションメモ: iOS 合成パイプラインのリファクタ

作成日: 2026-04-17
作業ブランチ: `feature/models-refactor`

PronounSE の iOS 実装 (`CoreMLAudioApp`) における合成パイプラインの可読性改善と、その過程で行った CoreML / Transformer の理解整理の記録。

---

## 1. きっかけ

### 出発点となった問題意識

- 録音時間が 11 秒で長すぎる
- 「フーリエ変換で最初の音声だけ切り取っている」状態（と感じていた）
- iOS での合成が体感的に異常に遅い

### コード調査でわかった現状

- `AudioRecorder.swift:7` の `maxDuration = 11.0` は CoreML モデルの入力上限 (1000 フレーム ≒ 11.6 秒) に対応した上限値
- `preprocess` は onset 検出で **「先頭の無音をトリム」** しているだけで、音声を最初の部分だけに切り取る処理ではない
- 遅さの正体は別にある（後述）

---

## 2. 「iOS が遅い」の原因分析

### 直接の原因: NSNumber ボクシング

`AudioSynthesizer.swift` の自己回帰デコードループで、毎ステップ **NSNumber を大量に生成** していた:

```swift
for i in 0..<(currentLength * nMels) {
    decInput[i] = NSNumber(value: decoderInputData[i])  // 1要素ごとに NSNumber 生成
}
```

- ステップ k では `k * 256` 回の NSNumber アロケーション
- 944 ステップ累計で **約 1.13 億回** の heap allocation
- これが Python (PyTorch) には存在しない iOS 固有のコスト

### NSNumber とは

Objective-C 時代の API で、`Float` などの数値を「オブジェクト」としてラップした型。Foundation のコレクションがオブジェクトしか持てないという制約から生まれた歴史的負債。

| 操作 | `Float` (生の値) | `NSNumber` |
|---|---|---|
| メモリ | スタック/レジスタ | ヒープに確保 |
| 確保コスト | ほぼゼロ | `malloc` 相当 |
| 寿命管理 | スコープを抜ければ消える | ARC の retain/release |
| メモリ局所性 | 連続して並ぶ | ヒープ上に散らばる |

### 補助的な要因

- **MainActor ホップ × 944 回**: 毎ステップ UI スレッドへスイッチ
- **CoreML の動的入力形状**: `currentLength` が毎回変わるため Neural Engine のキャッシュが効きにくい
- **ハードウェア差**: M4 Max と iPhone の生の計算能力差

### 解決策

`MLMultiArray.dataPointer` で生メモリに直接アクセスすれば NSNumber を回避できる。

```swift
let ptr = decInput.dataPointer.assumingMemoryBound(to: Float.self)
decoderInputData.withUnsafeBufferPointer { src in
    ptr.update(from: src.baseAddress!, count: count)
}
```

ただし今回は**実装せず**。可読性改善が先と判断。

---

## 3. 「これは研究になるか」の整理

純粋な NSNumber → dataPointer 置換は iOS エンジニアリングの定番テクニックで、それ自体は研究ではない。

研究化できる切り口:
- (a) **ボトルネック分析**: 「CoreML 推論時間 vs ホスト側マーシャリング時間」の内訳定量化
- (b) **既存研究方針との接続**: 量子化評価の総合レイテンシ測定で「実装オーバーヘッドを除去した上で」推論時間を比較
- (c) **自己回帰デコードの実装方式比較**: 動的形状 / 固定長+マスク / KV キャッシュ / バッチ推論

学部研究のスコープとしては **(b)** が現実的。

### モデル構造側の改善余地

`scripts/convert_transformer.py` の `RangeDim(1, 1000)` を固定形状に変えると Neural Engine 利用率が上がる可能性。これは Phase 2 のボトルネック分析の素材になる。

Stateful CoreML (KV キャッシュ) は技術的に魅力的だが学部スコープを超える。

---

## 4. 方針転換: 可読性優先のリファクタへ

性能改善より先に **「`AudioSynthesizer.swift` の責務を小さくして読みやすくしたい」** という根本的な要望が判明。

さらに **「リファクタを通じて実装を理解したい」** という学習目的も合わせて、慎重に段階的に進める方針に切り替え。

---

## 5. 実装した変更（コミット履歴）

ブランチ `feature/models-refactor` に 6 コミット。

### コミット 110f73c: ディレクトリ整理

平らに並んでいた 9 ファイルを 4 カテゴリのサブディレクトリに分割:

```
Models/
├── Audio/              ← 録音・再生・特徴量
│   ├── AudioRecorder.swift
│   ├── AudioPlayer.swift
│   ├── AudioSource.swift
│   └── AudioFeatureExtractor.swift
├── Synthesis/          ← 合成パイプライン
│   ├── AudioSynthesizer.swift
│   └── SynthesisResult.swift
├── Configuration/      ← UI 設定
│   ├── ComputeUnitOption.swift
│   └── ModelPrecision.swift
└── Debug/              ← デバッグ
    └── DebugStats.swift
```

Xcode 16 の synchronized groups を使っているため pbxproj 編集不要。`git mv` で履歴も追える。

### コミット d4712ad: ArrayStats.compute(from:) に集約

`AudioSynthesizer.swift` 内の private ヘルパ `computeStats(of:)` 2 つ（合計 40 行）を、`ArrayStats` 型の static factory メソッドとして `Debug/DebugStats.swift` に切り出し。

```swift
// Before
let encoderStats = computeStats(of: memory)

// After
let encoderStats = ArrayStats.compute(from: memory)
```

呼び出し側 7 箇所も置換。

#### computeStats の正体

git 履歴を辿った結果、これは `DebugStatsView` のために導入された統計計測ヘルパだと判明（コミット e1da01d で同時追加された）。**合成本体ロジックではなく、観察用の道具**だったため Debug/ に移すのが自然。

### コミット eec80ba: EncoderRunner の切り出し

Encoder 実行ロジック（22 行）を `Synthesis/EncoderRunner.swift` に分離。

```swift
// AudioSynthesizer.synthesize() の中
let encoderRunner = EncoderRunner(model: encoder)
let memory = try await encoderRunner.run(mel: melData, frameCount: frameCount, nMels: nMels)
let encoderStats = ArrayStats.compute(from: memory)
```

設計判断:
- Runner は推論実行のみに専念、デバッグ統計は呼び出し側で計算
- Runner はステートレスに近い薄い wrapper として `synthesize()` のたびに生成
- `loadModels` は変更せず、MLModel の保持は AudioSynthesizer 側のまま

### コミット f634c87: VocoderRunner の切り出し

HiFi-GAN 実行ロジック（約 30 行）を `Synthesis/VocoderRunner.swift` に分離。

設計上の重要な気づき: **min/max/mean/hasNaN/hasInf は要素の並び順に依存しない**。

これにより、`hifiganInputStats` を **転置前の `postnetOut`** で計算できるようになり、転置処理を Runner の中に隠蔽できた。

```swift
// 入力統計は転置前で計算（値は転置後と同値）
let hifiganInputStats = ArrayStats.compute(from: postnetOut)

let vocoderRunner = VocoderRunner(model: hifigan)
var waveform = try await vocoderRunner.run(postnetOut: postnetOut, totalFrames: totalFrames, nMels: nMels)
```

### コミット c2a29ee: CoreML API 学習ノート追加

`docs/coreml-api-notes.md` に CoreML の典型パターンと冗長性の背景を整理。

### コミット 59354a2: DecoderRunner の切り出し

Decoder 自己回帰ループ（約 65 行）を `Synthesis/DecoderRunner.swift` に分離。最複雑なため最後にまわした。

設計判断:
- 状態管理 (`decoderInputData`, `currentLength`)、ループ、ステップ統計の選別記録、`Task.yield()` をすべて Runner 内に閉じ込める
- 進捗は Runner からは「N ステップ目完了」とだけ通知 (`onStep: (Int) -> Void`)、パーセント計算 (`0.1 + 0.8 * step/frameCount`) は呼び出し側
- 戻り値は `(postnetOut, stepStats)` の tuple
- ステップ統計だけは Runner 内で完結（ループに本質的に紐づくため）。Encoder/Vocoder の方針 (stats は呼び出し側) と異なる

```swift
// AudioSynthesizer.synthesize() の中
let decoderRunner = DecoderRunner(model: decoder)
let (postnetOut, decoderStepStats) = try await decoderRunner.run(
    memory: memory,
    frameCount: frameCount,
    nMels: nMels,
    onStep: { completed in
        let progress = 0.1 + 0.8 * Double(completed) / Double(frameCount)
        onProgress("Decoder 実行中... (\(completed)/\(frameCount))", progress)
    }
)
```

---

## 6. リファクタ完了後の synthesize() の構造

```
1. メルスペクトログラム抽出   ← AudioFeatureExtractor (既存)
2. Encoder                   ← EncoderRunner (新)
3. Decoder (自己回帰ループ)  ← DecoderRunner (新)
4. HiFi-GAN                  ← VocoderRunner (新)
5. デエンファシス            ← AudioFeatureExtractor (既存)
6. SynthesisResult 組み立て
```

`AudioSynthesizer.swift` は当初 285 行 → 約 150 行（**約 47% 削減**）。

合成パイプラインの**論理フローだけが見える**ようになった。

---

## 7. 学習内容のサマリ

詳細は `docs/coreml-api-notes.md` 参照。

### EncoderRunner.run() で起きていること（4 ステップ）

1. **入力テンソルを作る**: `[Float]` を `MLMultiArray` に詰め直し
2. **位置情報テンソルを作る**: `[1, 2, ..., T]` の Int32 配列（Transformer 特有）
3. **入力辞書に包んで predict**: `MLDictionaryFeatureProvider` に名前付きで渡す
4. **出力から memory を取り出す**: `output.featureValue(for:).multiArrayValue`

### CoreML 特有 vs Transformer 特有

| 要素 | CoreML 特有 | Transformer 特有 |
|---|:---:|:---:|
| MLMultiArray, NSNumber, MLFeatureValue | ✅ | |
| 辞書化、Optional 取り出し | ✅ | |
| `pos` 配列 (1, 2, ..., T) | | ✅ |
| 入力が mel + pos の 2 つ | △ | ✅ |

PyTorch なら 2 行（`memory = encoder(mel, pos)`）の処理が CoreML だと約 15 行に膨らむ。**膨らんだ部分の大半が CoreML 由来の儀式**。

### なぜラップだらけか

5 段ラップ (`Float` → `NSNumber` → `MLMultiArray` → `MLFeatureValue` → `MLDictionaryFeatureProvider`) は、**画像 / 物体検出 / TTS / テキスト分類すべてを 1 つの API で扱う汎用性のために犠牲にした単純さ**。Apple 自身も認識して iOS 16 で `MLShapedArray<Float>` を追加。

### なぜ転置が必要か

Transformer と CNN で軸の慣習が違う:

| | shape | 由来 |
|---|---|---|
| Transformer | `[batch, time, feature]` | フレーム単位の処理が自然 |
| CNN (Conv1d) | `[batch, channel, time]` | 時間方向にカーネルを滑らせるのが自然 |

Decoder (Transformer) → HiFi-GAN (CNN) の境界で形を合わせる必要がある。これは PronounSE 固有ではなく、Transformer + CNN の 2 段モデル全般で発生する（Tacotron + WaveNet 等）。

PyTorch なら `t.transpose(x, 1, 2)` の 1 行。CoreML には簡便な API がないので二重ループで書く。

---

## 8. 残作業

### このブランチ内の続き

- ビルド確認: Xcode で開いて全コミットの動作確認

### 別タスクとして残している

- **NSNumber → dataPointer の置換** (性能改善): リファクタ完了後に DecoderRunner 内で局所化して実施
- **録音時間の短縮** (11s → 3s): 1 行変更だが UI 整合確認が必要
- **モデル変換の固定形状化** (`RangeDim` → 固定): 研究テーマと直結する Phase 2 素材

### 研究との接続

- 今回の整理により、Phase 1「段別レイテンシ計測」を入れる場所が明確化（各 Runner の入口/出口）
- 「CoreML 化で PyTorch 2 行が 15 行に膨らむ」観察は論文の「オンデバイス化の実装コスト」章の素材

---

## 9. 関連ドキュメント

- `docs/research-planning.md` — 研究方針メモ（4 フェーズ構成）
- `docs/coreml-api-notes.md` — CoreML API の詳細解説
- `.claude/rules/git-workflow.md` — Git 運用ルール
- `.claude/rules/ios-app.md` — iOS アプリの構造規約
