# CoreML のラップ・型変換発生量の集計

iOS アプリ (`CoreMLAudioApp`) の合成パイプラインで発生している、
CoreML 固有のラップ・型変換・Optional アンラップを網羅的に洗い出したメモ。

「CoreML の書き方は回りくどい」という体感を定量化するのが目的。
研究の主題ではなく**実装上の知見**として記録する。

---

## 1. 発生しているラップ・型変換の全種類

| # | 何を何に変換してる？ | 書き方 | なぜ必要？ |
|---|---|---|---|
| ① | **Float → NSNumber** | `NSNumber(value: floatValue)` | MLMultiArray に Float 値を入れるため（Objective-C の配列仕様） |
| ② | **Int → NSNumber** | `86 as NSNumber` | shape 指定やインデックス指定で必要 |
| ③ | **[Float] → MLMultiArray** | `MLMultiArray(shape:..., dataType:.float32)` + for ループ代入 | CoreML モデルが受け取る型 |
| ④ | **MLMultiArray → MLFeatureValue** | `MLFeatureValue(multiArray: arr)` | モデル入力用に「包む」 |
| ⑤ | **辞書 → MLDictionaryFeatureProvider** | `MLDictionaryFeatureProvider(dictionary: [...])` | モデルに渡す最終形式 |
| ⑥ | **MLFeatureProvider → MLMultiArray（出力）** | `output.featureValue(for: "...")?.multiArrayValue` | 出力辞書から取り出す（Optional 2 重） |
| ⑦ | **NSNumber → Float** | `arr[i].floatValue` | MLMultiArray の値を Swift で使うため |
| ⑧ | **MLMultiArray → [Float]** | for ループで ⑦ を count 回繰り返す | Swift 標準配列に戻す |

8 種類のラップ・変換が入り混じっている。

---

## 2. 1 秒の音声を合成するときの実際の回数

前提: `frameCount ≈ 86`、`nMels = 256`、波形 22050 サンプル。

| 発生箇所 | 変換種類 | 回数 |
|---|---|---|
| **Encoder** 入力 mel の NSNumber 詰め | Float → NSNumber | **22,016 回** |
| Encoder 入力 pos | Int32 → NSNumber | 86 回 |
| **Decoder** 入力 decoder_input の NSNumber 詰め（全ループ合計） | Float → NSNumber | **約 957,696 回** |
| Decoder 入力 pos（全ループ合計） | Int32 → NSNumber | 3,741 回 |
| Decoder 出力から次入力へ append（全ループ合計） | NSNumber → Float | 22,016 回 |
| **Vocoder** 転置のインデックス変換 | Int → NSNumber（× 4 / イテレーション） | 88,064 回 |
| Vocoder 出力波形の NSNumber → Float | NSNumber → Float | 22,016 回 |
| 出力メル取り出し（AudioSynthesizer） | NSNumber → Float | 22,016 回 |
| **合計** | | **約 114 万回** |

1 秒の音声を作るだけで 100 万回以上の NSNumber ラップ / アンラップが走る。
これが verbose の体感の正体。

スケールの目安:

| 音の長さ | NSNumber ラップ・アンラップ回数 |
|---|---|
| 1 秒 | 約 114 万回 |
| 5 秒 | 約 570 万回 |
| 10 秒 | 約 1,140 万回 |

（※ Decoder の自己回帰ループが二乗オーダーで増える部分があるので、
  長くなるほど増え方は加速する）

---

## 3. ファイル別の発生箇所

| ファイル | 発生している変換 | コード行 |
|---|---|---|
| `EncoderRunner.swift` | ①②③④⑤⑥ | 22-44 |
| `DecoderRunner.swift` (ループ内) | ①②③④⑤⑥⑦ | 35-80 |
| `VocoderRunner.swift` | ②③④⑤⑥⑦⑧ | 23-45 |
| `AudioSynthesizer.swift` (出力メル取り出し) | ②⑦ | 110-115 |

7 種類以上のラップがどのファイルにも出てくる。
共通パターンだが現時点ではヘルパー化されていない。

---

## 4. なぜそんなに多いのか（元凶の整理）

| 元凶 | 何が起きるか | 代替 |
|---|---|---|
| **MLMultiArray の要素型が NSNumber 固定** | Float を 1 個入れるのに毎回 `NSNumber(value:)` でラップ | `MLShapedArray<Float>` に移行すれば不要 |
| **shape を `[NSNumber]` で要求** | `Int` を毎回 `as NSNumber` キャスト | `MLShapedArray<Float>` なら `[Int]` でよい |
| **出力が辞書形式** | 毎回 `featureValue(for: "キー")` で取り出す | モデル変換時に出力型を型安全にすれば、自動生成クラスで `.memory` などアクセス可 |
| **Optional 2 重** | `?.multiArrayValue` でアンラップ地獄 | 同上 |

根本原因は 2 つ:

1. **Objective-C 時代の API 設計**（NSNumber 必須、Optional 多用）
2. **CoreML の汎用性優先**（画像・配列・テキストなど何でも扱える代わりに型が緩い）

---

## 5. パフォーマンス影響は？

1 回あたりの NSNumber ラップは数百ナノ秒オーダー。
100 万回でも数百ミリ秒〜1 秒弱のオーバーヘッド。

実測が必要だが、**合成全体の時間に対しては誤差レベルの可能性が高い**。
verbose の実害は「実行時間」ではなく「コード量・認知コスト・保守性」。

---

## 6. 改善の方向（未実施）

- **ヘルパー拡張を作る**（`MLMultiArray.from(floats:shape:)`、`MLMultiArray.toFloatArray()` など）
- **`MLShapedArray<Float>` ベースで `.mlpackage` を再変換**する
- **出力クラスの自動生成**を使って辞書アクセスを型安全にする

いずれも「書きやすさの改善」であり、本研究の主題（量子化・オンデバイス化）
ではない。研究発表での主張の軸にはしない。必要に応じて実装ノートに 1〜2 文で触れる程度の位置づけ。

---

## 関連

- [`decoder-runner-breakdown.md`](decoder-runner-breakdown.md) — DecoderRunner.swift の読み方（T/C/U/D 分類）
- [`../2026-04-17/coreml-api-notes.md`](../2026-04-17/coreml-api-notes.md) — CoreML API の典型パターン整理（EncoderRunner 題材）
