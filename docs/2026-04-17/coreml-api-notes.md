# CoreML API 学習ノート

作成日: 2026-04-17

iOS アプリで PronounSE を動かす過程で、CoreML の API がなぜこういう形をしているのかを理解するためのメモ。Transformer Encoder の切り出し (`EncoderRunner.swift`) を題材に整理した。

---

## 1. EncoderRunner.run() がやっていること

`EncoderRunner.swift` の `run` メソッド全体。CoreML モデルを 1 回叩くための典型パターンが詰まっている。

```swift
func run(mel: [Float], frameCount: Int, nMels: Int) async throws -> MLMultiArray {
    // ① 入力テンソルを作る
    let melArray = try MLMultiArray(shape: [1, frameCount as NSNumber, nMels as NSNumber], dataType: .float32)
    for i in 0..<(frameCount * nMels) {
        melArray[i] = NSNumber(value: mel[i])
    }

    // ② 位置情報テンソルを作る
    let posArray = try MLMultiArray(shape: [1, frameCount as NSNumber], dataType: .int32)
    for i in 0..<frameCount {
        posArray[i] = NSNumber(value: Int32(i + 1))
    }

    // ③ 入力辞書に包んで predict
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "mel": MLFeatureValue(multiArray: melArray),
        "pos": MLFeatureValue(multiArray: posArray)
    ])
    let output = try await model.prediction(from: input)

    // ④ 出力辞書から memory を取り出す
    guard let memoryFeature = output.featureValue(for: output.featureNames.first ?? ""),
          let memory = memoryFeature.multiArrayValue else {
        throw AudioSynthesizer.SynthesisError.decoderFailed
    }
    return memory
}
```

### 全体の流れ

```
Swift 配列 [Float]
    ↓ ① MLMultiArray に詰め直す
MLMultiArray (mel) + MLMultiArray (pos)
    ↓ ② キー名付き辞書に包む
MLDictionaryFeatureProvider
    ↓ ③ predict を await
MLFeatureProvider (出力辞書)
    ↓ ④ 出力辞書から取り出す
MLMultiArray (memory)
```

---

## 2. ① 入力テンソルを作る

```swift
let melArray = try MLMultiArray(shape: [1, frameCount as NSNumber, nMels as NSNumber], dataType: .float32)
for i in 0..<(frameCount * nMels) {
    melArray[i] = NSNumber(value: mel[i])
}
```

### shape の意味

3 次元テンソル `[バッチサイズ, 時間方向, メル次元]`:

```
バッチ = 1（1 個の音声）
時間方向 = frameCount（例: 100 フレーム ≒ 1.16 秒）
メル次元 = 256（周波数ビン）
```

### `as NSNumber` が必要な理由

`MLMultiArray(shape:dataType:)` の `shape` は `[NSNumber]` を要求する。Objective-C 時代の API なので数値もオブジェクトに包む必要がある。

### 1 次元ループでコピーできる理由

`mel` は 3 次元データを行優先で平らに並べたもの。MLMultiArray も内部的に同じ並びでメモリを確保しているので、1 次元の `i` を 1 対 1 で対応させてコピーできる。

---

## 3. ② 位置情報テンソルを作る (Transformer 特有)

```swift
let posArray = try MLMultiArray(shape: [1, frameCount as NSNumber], dataType: .int32)
for i in 0..<frameCount {
    posArray[i] = NSNumber(value: Int32(i + 1))
}
```

### なぜ pos が必要なのか

Transformer は self-attention を使って **全フレームを並列に処理** する。順番という概念を内蔵していない。なので「いま何番目のフレームか」を別途明示する必要がある。

```
posArray = [1, 2, 3, 4, ..., frameCount]
```

これは CoreML だろうと PyTorch だろうと TensorFlow だろうと、**Transformer なら必ず必要**。

### なぜ 1 始まりなのか

PronounSE のオリジナル Python (`synthesis.py:54`) が `np.arange(1, mel.shape[0] + 1)` で 1 始まり。**0 はパディング用** として予約する慣習。

### RNN だったら不要

仮に PronounSE が Tacotron みたいな RNN ベースなら pos は不要。RNN は系列を順番に処理するので順序情報を内蔵している。

---

## 4. ③ 入力辞書を作って predict

```swift
let input = try MLDictionaryFeatureProvider(dictionary: [
    "mel": MLFeatureValue(multiArray: melArray),
    "pos": MLFeatureValue(multiArray: posArray)
])
let output = try await model.prediction(from: input)
```

### なぜ辞書？

CoreML モデルの入力には名前がついている。Encoder の場合、PyTorch から CoreML に変換するとき (`scripts/convert_transformer.py:108-112`) で:

```python
ct.TensorType(name="mel", shape=...),
ct.TensorType(name="pos", shape=...),
```

と 2 つの名前付き入力を定義した。だから Swift 側でも「どの名前にどのテンソルを渡すか」を明示する必要がある。

### なぜ MLFeatureValue で再ラップ？

CoreML は MLMultiArray 以外（画像、文字列、辞書）も扱える汎用 API。それらを 1 つの統一型で扱うために `MLFeatureValue` でラップする。

```
+-------------------+
| MLFeatureValue    |  ← 共通の包み紙
|  └─ MLMultiArray  |  ← 中身（画像でも文字列でも可）
+-------------------+
```

### predict の正体

```swift
let output = try await model.prediction(from: input)
```

これがモデルにデータを通す唯一の処理。

```
入力辞書 {"mel": ..., "pos": ...}
        ↓
[ Encoder ニューラルネットワーク ]
        ↓
出力辞書 {"some_output_name": ...}
```

`await` がついているのは、Neural Engine や GPU で実行するときに非同期処理になり得るため。

---

## 5. ④ 出力から memory を取り出す

### memory とは何か

Transformer Encoder が生成する中間表現:

```
入力メル    [1, T, 256]
   ↓
Encoder (= 多層の self-attention)
   ↓
memory      [1, T, 512]   ← 各フレームを 512 次元のベクトルで表現したもの
```

「**Encoder の出力を Decoder が参照するもの**」を memory と呼ぶのが Transformer の用語。Decoder は生成のたびに「Encoder が記憶している memory のどこに注目するか」を attention で決める。

### 取り出すコード

```swift
guard let memoryFeature = output.featureValue(for: output.featureNames.first ?? ""),
      let memory = memoryFeature.multiArrayValue else {
    throw AudioSynthesizer.SynthesisError.decoderFailed
}
return memory
```

### 第 1 段階: キー名で値を引く

```swift
let memoryFeature = output.featureValue(for: "出力名")
```

`output.featureNames.first` を使っているのは Encoder の出力が 1 つだけだから。CoreML が変換時に自動命名するので（例: `var_123`）、その名前を直接書き下せない。

Decoder の場合は出力が 2 つ（mel_out と postnet_out）あるので `first` だと困る。

### 第 2 段階: 段ボール箱を開けて中身を取り出す

```swift
let memory = memoryFeature.multiArrayValue
```

`memoryFeature` は `MLFeatureValue` という箱なので、`.multiArrayValue` で「中身が `MLMultiArray` だったら取り出す」操作。中身が画像だったり文字列だったりすると `nil` が返るので Optional。

---

## 6. 入力と出力の対称性

```
{name: tensor, name: tensor}  ──(predict)──→  {name: tensor, name: tensor}
```

| | 入力 | 出力 |
|---|---|---|
| 容器 | `MLDictionaryFeatureProvider` | `MLFeatureProvider` |
| エントリ | `[名前: MLFeatureValue]` | `[名前: MLFeatureValue]` |
| 中身 | `MLMultiArray` | `MLMultiArray` |
| 操作 | キー名にテンソルを設定 | キー名でテンソルを取得 |

CoreML のモデル呼び出しは「**辞書 → 辞書の関数**」と思えばよい。

---

## 7. CoreML 特有 vs Transformer 特有

EncoderRunner の冗長さは、**2 種類の複雑さの層が重なっている** だけで、本来は別物。

### 各要素の出自

| コードの要素 | CoreML 特有 | Transformer 特有 | 備考 |
|---|:---:|:---:|---|
| `MLMultiArray` 型に詰め直す | ✅ | | PyTorch なら `torch.tensor` 一発 |
| `as NSNumber` のキャスト | ✅ | | Obj-C 遺産 |
| 要素ごとに NSNumber でループコピー | ✅ | | PyTorch は連続メモリを直接渡せる |
| `shape: [1, T, nMels]` という 3 次元 | | | どっちでもない（深層学習一般の慣習） |
| `pos` 配列 (1, 2, ..., T) を作る | | ✅ | Transformer の本質的要求 |
| 入力を `"mel"` `"pos"` の辞書にする | ✅ | △ | 入力が 2 つあるのは Transformer 由来、辞書化は CoreML 由来 |
| `MLFeatureValue` で再ラップ | ✅ | | CoreML の共通型システム |
| 出力辞書から `featureValue` で名前引き | ✅ | | PyTorch は `memory = model(...)` で済む |
| `multiArrayValue` の Optional 解除 | ✅ | | CoreML の API 設計 |

### PyTorch だと何行で書けるか

オリジナルの `PronounSE/synthesis.py:131-137` を見ると、同じ処理がたった 2 行:

```python
mel_src, pos_src = wav2feature(file_path)
memory, c_mask = m.procEncoder(mel_src, pos_src)
```

これを CoreML で書くと約 15 行に膨らむ。**膨らんだ部分の大半が CoreML 由来の儀式**。

### 「カオスに感じる」原因

```
+--------------------------------------+
|  Transformer 由来の要求              |  ← 薄い層（pos 配列くらい）
|  - 入力が mel + pos の 2 つ           |
+--------------------------------------+
|  CoreML 由来の API 儀式              |  ← 厚い層（大半）
|  - MLMultiArray の生成と詰め込み      |
|  - NSNumber へのキャスト              |
|  - 辞書化、Optional 取り出し          |
+--------------------------------------+
```

それぞれ単独なら整然としているが、**インラインで同居すると Transformer のロジックが CoreML の儀式の中に埋もれる**。これがカオスに見える正体。

リファクタの方針は **2 つの層を物理的に別ファイルに分離する** こと:

| ファイル | 何に集中するか |
|---|---|
| `EncoderRunner.swift` 等 | CoreML の儀式（1 段ぶん）をパッケージ化 |
| `AudioSynthesizer.swift` | Transformer の論理フロー（Encoder→Decoder→Vocoder）だけが見える |

---

## 8. なぜ CoreML はラップだらけなのか

5 段ラップの正体:

```
Float (生の数値)
  ↓ NSNumber (オブジェクト化)
  ↓ MLMultiArray (テンソル型)
  ↓ MLFeatureValue (汎用「特徴値」)
  ↓ MLDictionaryFeatureProvider (辞書プロバイダ)
```

### 各層の言い分

| 層 | 存在理由 |
|---|---|
| `NSNumber` | Foundation コレクションがオブジェクトしか持てない (Obj-C 遺産) |
| `MLMultiArray` | 複数の数値型 (Float32/16, Int32...) と任意次元を 1 型で扱う |
| `MLFeatureValue` | 画像/文字列/辞書/マルチアレイなど何でも入る共通の入れ物 |
| `MLDictionaryFeatureProvider` | 入力に名前を付けて複数渡せる |

### 結論: 汎用性のために犠牲にした単純さ

CoreML は **画像認識 / 物体検出 / 音声合成 / テキスト分類 / etc. のすべてを 1 つの API で扱う** ために抽象化を重ねている。それぞれの抽象化には理由があるが、「ただの Float テンソルを渡したい」だけのユースケースには過剰。

### 例え

> 牛乳 1 本を買うのに、「液体である」「容器に入っている」「飲料である」「乳製品である」「商品である」と 5 段階の証明書を毎回書いている状態

### Apple 自身も「やりすぎた」と認識した

iOS 16 (2022) で **`MLShapedArray<Float>`** という Swift ネイティブの新型が追加された:

```swift
// 旧: 5 段ラップ
let arr = try MLMultiArray(shape: [1, 100, 256], dataType: .float32)
arr[0] = NSNumber(value: 1.5)
let feature = MLFeatureValue(multiArray: arr)
let provider = try MLDictionaryFeatureProvider(dictionary: ["mel": feature])

// 新: Swift ネイティブ
let arr = MLShapedArray<Float>(scalars: [...], shape: [1, 100, 256])
```

Apple 自身が「あの設計はカジュアル用途には冗長すぎた」と認めて新 API を追加した。既存コードのために古い API も残している。

### 他フレームワークとの比較

| フレームワーク | コード行数 |
|---|---|
| **PyTorch** | 1〜2 行 (`memory = encoder(mel, pos)`) |
| **TensorFlow Lite** | 5〜8 行 |
| **ONNX Runtime** | 5〜10 行 |
| **CoreML** | **15〜20 行** |

CoreML が突出して冗長。**型安全 + 動的検査 + 汎用性** を優先して **書きやすさ** を犠牲にした設計判断の結果。

---

## 9. 研究との接続

`docs/research-planning.md` の Phase 2「失敗原因の究明」や論文の考察で使える観察:

> PyTorch 2 行が CoreML 15 行に膨らむ。**この変換コストが、PyTorch ベースの研究プロトタイプを iOS で動かす際の障壁の一つ** として観測された。

オンデバイス化の現場で実際にぶつかる **API 設計上の摩擦** を記述するのは、エンジニアリング感覚のある研究としての側面を強化できる素材。

---

## 10. まとめ

| 質問 | 答え |
|---|---|
| なぜラップだらけ？ | 汎用性 + 歴史的経緯 + 型安全志向の合作 |
| 必然なのか？ | 設計判断の結果。技術的必然ではない |
| 改善の余地は？ | iOS 16 で `MLShapedArray` という新 API が出た |
| 「冗長に感じる」は正しい？ | 正しい。Apple の API がそういうもの |

選択肢:
- 受け入れて慣れる
- `MLShapedArray` で書き直す
- `dataPointer` で escape hatch を使う（性能対策にも有効）
