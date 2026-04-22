# DecoderRunner.swift の読み方

`ios/CoreMLAudioApp/CoreMLAudioApp/Models/Synthesis/DecoderRunner.swift` は
「Transformer Decoder の自己回帰ループ」を CoreML で動かすクラス。

このドキュメントは**コードを 4 種類に分けて読む**ための整理メモ。
Transformer の本質は 5 行くらいしかなく、残りは CoreML / UI / デバッグのお作法。
混ざって読むと認知コストが高くなるので、色分けして読む。

---

## 1. Transformer Decoder の本質（擬似コード 4 行）

CoreML でも PyTorch でもフレームワーク関係なく、Transformer Decoder
の自己回帰ループは本質的にこれだけ:

```
decoder_input = [ゼロフレーム]   # 長さ 1 から始める

for step in 0..<frame_count:
    mel_out, postnet = decoder(memory, decoder_input)
    decoder_input.append(mel_out の最後のフレーム)   # ← 自己回帰

return postnet   # 最後のループで作った postnet
```

- `memory` は Encoder が作った参考資料（ループ中は変わらない）
- `decoder_input` はループごとに 1 フレームずつ伸びる
- 最後のループで出てきた `postnet` を返す

研究発表で説明すべきはこの 4 行の論理。残りは実装詳細。

---

## 2. 用語の直感

### memory（Encoder の出力）

Encoder が入力音声を読んで作った「時間フレームごとの特徴ベクトル」。
形は `[1, T_src, 512]`。Decoder は自己回帰ループの毎回この memory を
参照する。だから「ずっと参照される記憶 = memory」と呼ぶ。

### 自己回帰（autoregressive）

1 フレームずつ順番に出力を作る方式。各ループで「今まで作った出力全部 +
memory」を入力にして、次の 1 フレームを予測する。

音のフレームは前後のつながりが重要（前の音に自然に続く必要がある）。
一気に全部生成すると自然さが出ないので、順番に作る。

ループ回数の例:

| 音の長さ | ループ回数 |
|---|---|
| 1 秒 | 約 86 回 |
| 5 秒 | 約 430 回 |
| 10 秒 | 約 860 回 |

Transformer 推論が遅い最大の理由はこのループ。

### Attention

「今、入力のどこに注目すべきか」を自動で計算する仕組み。

たとえば Decoder が「出力フレーム 10 を作ろう」と考えているとき、
memory の中の各フレームに「注目度（%）」を計算し、重みつきで
参照する。注目度は学習で決まる（人間が指定するわけではない）。

Transformer の中身はこの Attention 計算の積み重ね。

---

## 3. コードの色分け

DecoderRunner のループ本体を 4 種類に分けて読む:

- **【T】Transformer の本質** — モデルのアルゴリズム
- **【C】CoreML のお作法** — テンソル操作、モデル呼び出し
- **【U】iOS UI のお作法** — 進捗通知、スレッド切り替え
- **【D】デバッグ用** — なくても動く、NaN/Inf 検出のため

```swift
for step in 0..<frameCount {                                              // 【T】自己回帰ループ本体

    // ──────── 【C】: 入力テンソルを作る ────────
    let decInput = try MLMultiArray(shape: [1, currentLength, nMels],     // 【C】
                                     dataType: .float32)                   // 【C】
    for i in 0..<(currentLength * nMels) {                                 // 【C】フラット配列→テンソル
        decInput[i] = NSNumber(value: decoderInputData[i])                 // 【C】NSNumber でラップ必須
    }

    let decPos = try MLMultiArray(shape: [1, currentLength],               // 【C】位置情報テンソル
                                   dataType: .int32)                       // 【C】
    for i in 0..<currentLength {                                           // 【C】
        decPos[i] = NSNumber(value: Int32(i + 1))                          // 【C】
    }

    let input = try MLDictionaryFeatureProvider(dictionary: [              // 【C】辞書形式で入力
        "memory":        MLFeatureValue(multiArray: memory),               // 【C】
        "decoder_input": MLFeatureValue(multiArray: decInput),             // 【C】
        "pos":           MLFeatureValue(multiArray: decPos)                // 【C】
    ])

    // ──────── 【T】のメイン: Decoder 1 回分の予測 ────────
    let output = try await model.prediction(from: input)                   // 【T】decoder(...) の本体

    // ──────── 【C】: 出力辞書から値を取り出す ────────
    let postnetKey = output.featureNames.first(where: { $0 != "mel_out" }) ?? ""  // 【C】
    lastMelOut     = output.featureValue(for: "mel_out")?.multiArrayValue          // 【C】
    lastPostnetOut = output.featureValue(for: postnetKey)?.multiArrayValue         // 【C】
    guard let melOut = lastMelOut else { throw ... }                               // 【C】Optional 処理

    // ──────── 【D】: デバッグ統計 ────────
    let melStats = ArrayStats.compute(from: melOut)                        // 【D】
    let postStats = lastPostnetOut.map { ... } ?? ...                      // 【D】
    let shouldRecord = step < 5 || step >= frameCount - 5 || ...           // 【D】先頭/中間/末尾を記録
    if shouldRecord {                                                       // 【D】
        stepStats.append(DecoderStepStats(...))                             // 【D】
    }

    // ──────── 【T】: 自己回帰の核心（次の入力に追加） ────────
    for i in 0..<nMels {                                                    // 【T】mel_out の最後のフレームを
        decoderInputData.append(                                            // 【T】decoderInputData に
            melOut[[0, (currentLength - 1) as NSNumber, i as NSNumber]]     // 【T】append する
                  .floatValue                                                // 【T】（これが自己回帰）
        )
    }
    currentLength += 1                                                      // 【T】長さを 1 増やす

    // ──────── 【U】: UI のためのお作法 ────────
    await MainActor.run { onStep(step + 1) }                                // 【U】進捗通知
    if step % 10 == 0 { await Task.yield() }                                // 【U】CPU を譲る
}

guard let postnetOut = lastPostnetOut else { throw ... }                    // 【C】Optional 処理
return (postnetOut, stepStats)                                               // 【T】最終的な出力を返す
```

---

## 4. 行数の内訳

| 種別 | 行数 | 割合 |
|---|---|---|
| 【T】Transformer の本質 | 約 5 行 | 20% |
| 【C】CoreML のお作法 | 約 15 行 | 50% |
| 【U】UI のお作法 | 約 3 行 | 10% |
| 【D】デバッグ用 | 約 5 行 | 20% |

**Transformer として意味のあるコードは全体の 2 割**。残り 8 割はお作法・デバッグ。

---

## 5. 研究発表で説明する時のテンプレ

DecoderRunner.swift の中身は 3 つの層に分かれている:

1. **Transformer の自己回帰ループ**（本質、5 行）
2. **CoreML API にデータを橋渡しするコード**（iOS 実装詳細、15 行）
3. **UI 連携とデバッグ計測のコード**（ユーザー体験と診断、8 行）

研究的には 1 が主役、2 は「CoreML に合わせた実装」と言えば十分。

---

## 関連

- Python 側の対応実装: `PronounSE/Transformer/network.py` の `MelDecoder`
- Encoder 側の同等の整理: [`../2026-04-17/coreml-api-notes.md`](../2026-04-17/coreml-api-notes.md)
- リファクタ経緯: [`../2026-04-17/models-refactor.md`](../2026-04-17/models-refactor.md)
